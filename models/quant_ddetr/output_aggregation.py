import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# from .lsq_plus import *
# from ._quan_base_plus import truncation
from util import box_ops


# def symmetric_bce(p, q, eps=1e-8):
#     p = torch.clamp(p, eps, 1-eps) # avoid log(0)
#     q = torch.clamp(q, eps, 1-eps)
#     term1 = -p * torch.log(q) - (1-p) * torch.log(1-q)
#     term2 = -q * torch.log(p) - (1-q) * torch.log(1-p)
#     return (term1 + term2).sum(dim=-1)


class OutAggregate(nn.Module):
    def __init__(self, num_classes, t_b=0.9, t_c=12):
        super().__init__()
        self.num_classes = num_classes
        self.t_b = t_b
        self.t_c = t_c

    def forward(self, bboxes, logits):
        # prob = logits.sigmoid()
        with torch.no_grad():
            n_q = bboxes.shape[1]
            # print(n_q)
            iou_matrix = []
            for i in range(len(bboxes)):
                iou_matrix.append(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(bboxes[i]),
                    box_ops.box_cxcywh_to_xyxy(bboxes[i])
                ))
            iou_matrix = torch.stack(iou_matrix)
            iou_matrix_gt_threshold = iou_matrix > self.t_b

            # sbce_matrix = symmetric_bce(prob.unsqueeze(2), prob.unsqueeze(1))
            # sbce_matrix_lt_threshold = sbce_matrix < self.t_c
            
            aggregation_mask = iou_matrix_gt_threshold
            aggregation_mask = aggregation_mask | aggregation_mask.transpose(1, 2) # to ensure mask is symmetric

            # calculate the transitive closure 
            adj = aggregation_mask.to(torch.float32)
            t = 0
            while t < n_q:
                new_adj = ((adj + torch.matmul(adj,adj)) > 1e-6).to(torch.float32)
                if torch.all(new_adj==adj):
                    break
                adj = new_adj
                t += 1
            aggregation_mask = adj

        aggregated_bboxes = (aggregation_mask @ bboxes) / (torch.sum(aggregation_mask, -1, keepdim=True) + 1e-6)
        # aggregated_prob = (aggregation_mask @ prob) / (torch.sum(aggregation_mask, -1, keepdim=True) + 1e-6)
        # aggregated_logits = torch.special.logit(aggregated_prob, eps=1e-6)
        # aggregated_logits = (aggregation_mask @ logits) / (torch.sum(aggregation_mask, -1, keepdim=True) + 1e-6)

        return aggregated_bboxes, logits, aggregation_mask
        

