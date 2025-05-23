# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy

import wandb
import torch
from torch.nn import functional as F
import torchvision.transforms as TcT
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

scaler = torch.cuda.amp.GradScaler()
def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def mec_loss(p, z, lamda_inv, order=4):
    p = F.normalize(p)
    z = F.normalize(z)
    c = p @ z.T
    c = c / lamda_inv
    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else:
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)
    return trace

# this is the weight of mec_loss 
def get_current_consistency_weight(current, rampup_length, top):
    if rampup_length == 0:
        return top
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return top * float(np.exp(-5.0 * phase * phase))


def train_hybrid(outputs, targets, k_one2many, criterion, lambda_one2many):
    # one-to-one-loss
    loss_dict = criterion(outputs, targets)
    multi_targets = copy.deepcopy(targets)
    # repeat the targets
    for target in multi_targets:
        target["boxes"] = target["boxes"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]

    # one-to-many loss
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if key + "_one2many" in loss_dict.keys():
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    k_one2many=1,
    lambda_one2many=1.0,
    use_wandb=False,
    use_fp16=False,
    use_mec=False,
    total_epochs=12,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        # pdb.set_trace()
        with torch.amp.autocast("cuda") if use_fp16 else torch.amp.autocast("cuda", enabled=False):
            if use_fp16:
                optimizer.zero_grad()

            outputs = model(samples)  # samples.tensors [2, 3, 800, 1066]  outputs["head_inputs"] [2, 17821, 256]

            if k_one2many > 0:
                loss_dict = train_hybrid(
                    outputs, targets, k_one2many, criterion, lambda_one2many
                )
            else:
                loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
       
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        if use_mec:
            # eps = 1e4   # 32
            warmup_epochs = 2
            top = 0.03
            it = len(data_loader) * (epoch - 1) + i
            total_mecloss = 0
            for head_input in outputs["head_inputs"]:
                b, C, H, W = head_input.shape  # [2, 256, 100, 134], [2, 256, 50, 67], [2, 256, 25, 34], [2, 256, 13, 17]
                d = C*H*W
                base_eps = 32
                base_d = 2048
                scale = d / base_d 
                eps = int(scale*base_eps)
                eps_d =  eps / d
                lamda = 1 / (b * eps_d)
                lamda_schedule = lamda_scheduler(8 / lamda, 1 / lamda, total_epochs, len(data_loader), warmup_epochs=warmup_epochs)
                lamda_inv = lamda_schedule[it]
                # print(lamda_inv)
                head_input = head_input.reshape(b, -1)
                mecloss = (-1) * mec_loss(head_input, head_input, lamda_inv) / b * lamda_inv / scale
                weight = get_current_consistency_weight(it, warmup_epochs*len(data_loader), top)
                total_mecloss += weight * mecloss
            losses += total_mecloss
            loss_dict["mec_loss"] = total_mecloss
    

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict) #distributed compute
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if use_fp16:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            optimizer.zero_grad()
            losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            )
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()

        if use_wandb:
            try:
                wandb.log(loss_dict)
            except:
                pass
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    distributed,
    use_wandb=False,
):
    # disable the one-to-many branch queries
    # save them frist

    # pdb.set_trace()
    if distributed:
        save_num_queries = model.module.num_queries
        save_two_stage_num_proposals = model.module.transformer.two_stage_num_proposals
        model.module.num_queries = model.module.num_queries_one2one
        model.module.transformer.two_stage_num_proposals = model.module.num_queries
    else:
        save_num_queries = model.num_queries
        save_two_stage_num_proposals = model.transformer.two_stage_num_proposals
        model.num_queries = model.num_queries_one2one
        model.transformer.two_stage_num_proposals = model.num_queries


    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    t=0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)

        # visualization
        if t % 100 == 0:
            # draw fig
            for batch, _result in enumerate(results):
                image = samples.tensors[batch]
                b = _result["boxes"]
                cb = _result["clustered_boxes"]
                cb_all = torch.concat(cb)
                draw_boxes_on_image(pil_transform_back(image), b.cpu(), bbox1_c="red", save_path=f"exps/vis/{t}_aggregated.png")
                draw_boxes_on_image(pil_transform_back(image), cb_all.cpu(), bbox1_c="cyan", save_path=f"exps/vis/{t}_unaggregated.png")
                draw_boxes_on_image(pil_transform_back(image), b.cpu(), cb_all.cpu(), save_path=f"exps/vis/{t}_all.png")
                for i in range(10):
                    # draw_boxes_on_image(pil_transform_back(image), boxes_gt[batch], bbox1_c="g", save_path=f"exps/vis/{t}_gt.png")
                    draw_boxes_on_image(pil_transform_back(image), b[i].unsqueeze(0).cpu(), bbox1_c="red", save_path=f"exps/vis/{t}_{i}_aggregated.png")
                    draw_boxes_on_image(pil_transform_back(image), cb[i].cpu(), bbox1_c="cyan", save_path=f"exps/vis/{t}_{i}_unaggregated.png")
                    draw_boxes_on_image(pil_transform_back(image), b[i].unsqueeze(0).cpu(), cb[i].cpu(), save_path=f"exps/vis/{t}_{i}_all.png")
        t += 1

        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    iou_aggregated = torch.mean(criterion.iou_aggregated)
    iou_unaggregated = torch.mean(criterion.iou_unaggregated)
    print(iou_aggregated)
    print(iou_unaggregated)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]
    if use_wandb:
        try:
            wandb.log({"AP": stats["coco_eval_bbox"][0]})
            wandb.log(stats)
        except:
            pass

    # recover the model parameters for next training epoch
    if distributed:
        model.module.num_queries = save_num_queries
        model.module.transformer.two_stage_num_proposals = save_two_stage_num_proposals
    else:
        model.num_queries = save_num_queries
        model.transformer.two_stage_num_proposals = save_two_stage_num_proposals

    return stats, coco_evaluator


# visualization
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    对经过归一化的张量进行反归一化操作。

    Args:
        tensor (torch.Tensor): 要反归一化的张量，形状为 (C, H, W)。
        mean (list): 归一化时使用的均值，默认为 [0.485, 0.456, 0.406]。
        std (list): 归一化时使用的标准差，默认为 [0.229, 0.224, 0.225]。

    Returns:
        torch.Tensor: 反归一化后的张量。
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    tensor = tensor * std.view(-1, 1, 1)
    tensor = tensor + mean.view(-1, 1, 1)

    return tensor


def pil_transform_back(image_tensor, resize=1280):
    """
    将经过ToTensor和Normalize变换后的张量转换回PIL图像。

    Args:
        image_tensor (torch.Tensor): 经过变换的图像张量，形状为 (C, H, W)。

    Returns:
        PIL.Image.Image: 恢复后的PIL图像。
    """
    unnormalized_tensor = unnormalize(image_tensor)

    # 将张量的值范围从 [0, 1] 转换到 [0, 255] 并转换为无符号8位整数类型
    pil_image = TcT.Resize(resize)(TcT.ToPILImage()(unnormalized_tensor.clamp(0, 1).mul(255).byte()))

    return pil_image


def draw_boxes_on_image(image, pred_bbox1=None, pred_bbox2=None, bbox1_c="cyan", bbox2_c="red", save_path=".draw_image.png"):
    """
    将给定的两个不同的边界框序列分别用不同颜色绘制到图像上，并保存到指定位置。

    Args:
        image (PIL.Image.Image): 要绘制边界框的PIL图像。
        pred_bbox1 (torch.Tensor): 第一个预测的边界框张量，形状为 [n_q, 4]，格式为 [x, y, w, h]，数值在 (0, 1) 表示图像尺寸的百分比。
        pred_bbox2 (torch.Tensor): 第二个预测的边界框张量，形状为 [n_q, 4]，格式为 [x, y, w, h]，数值在 (0, 1) 表示图像尺寸的百分比。
        save_path (str): 保存绘制好边界框图像的本地路径。

    Returns:
        None
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    width, height = image.size

    # 绘制第一个方框序列，用红色
    if pred_bbox1 is not None:
        for box in pred_bbox1:
            x_1, y_1, x_2, y_2 = box.tolist()
            x_pixel = int((x_1+x_2)/2)
            y_pixel = int((y_1+y_2)/2)
            w_pixel = int((x_2-x_1)/2)
            h_pixel = int((y_2-y_1)/2)
            rect = patches.Rectangle((x_pixel-w_pixel, y_pixel-h_pixel), 2*w_pixel, 2*h_pixel, linewidth=1, edgecolor=bbox1_c, facecolor='none')
            ax.add_patch(rect)

    # 绘制第二个方框序列，用蓝色
    if pred_bbox2 is not None:
        for box in pred_bbox2:
            x_1, y_1, x_2, y_2 = box.tolist()
            x_pixel = int((x_1+x_2)/2)
            y_pixel = int((y_1+y_2)/2)
            w_pixel = int((x_2-x_1)/2)
            h_pixel = int((y_2-y_1)/2)
            rect = patches.Rectangle((x_pixel-w_pixel, y_pixel-h_pixel), 2*w_pixel, 2*h_pixel, linewidth=1, edgecolor=bbox2_c, facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')

    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close()