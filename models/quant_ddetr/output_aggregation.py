import torch
from torch import nn
import torch.nn.functional as F

# from .lsq_plus import *
from ._quan_base_plus import truncation
from util import box_ops


def masked_kl_divergence(pred_logits: torch.Tensor, mask: torch.tensor) -> torch.Tensor:
    result_matrix = torch.full_like(mask.to(torch.float), float('inf'))
    mask_indices = torch.nonzero(mask, as_tuple=True)

    if len(mask_indices[0]) > 0:
        batch_idx, proposal_indices_row, proposal_indices_column = mask_indices

        relevant_pred_logits_row = pred_logits[:, proposal_indices_row]
        relevant_pred_logits_column = pred_logits[:, proposal_indices_column]

        log_softmax_preds_row = F.log_softmax(relevant_pred_logits_row, dim=1)
        softmax_preds_column = F.softmax(relevant_pred_logits_column, dim=1)

        kl_divs = F.kl_div(
            log_softmax_preds_row.unsqueeze(1),
            softmax_preds_column.unsqueeze(0),
            reduction='none'
        )
        result_matrix[batch_idx, proposal_indices_row, proposal_indices_column] = kl_divs.mean(dim=-1)

    return result_matrix


class OutAggregate(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, bboxes, logits):
        bboxes_q = truncation(bboxes, 4)[0]
        logits_q = truncation(torch.sigmoid(logits), 4)[0]

        # bboxs_q: [N, S, 4]
        # logits_q: [N, S, num_classes+1]
        iou_matrix = []
        for i in range(len(bboxes_q)):
            iou_matrix.append(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(bboxes_q[i]),
                box_ops.box_cxcywh_to_xyxy(bboxes_q[i])
            ))
        iou_matrix = torch.stack(iou_matrix)
        
        eye_boolean = torch.eye(iou_matrix.shape[1], dtype=torch.bool).to(iou_matrix.device)
        iou_matrix_gt_threshold = (iou_matrix > 0.9) ^ eye_boolean.unsqueeze(0)
        kl_div_matrix_lt_threshold = masked_kl_divergence(logits_q, iou_matrix_gt_threshold) < 0.3
        aggregation_mask = (iou_matrix_gt_threshold * kl_div_matrix_lt_threshold + eye_boolean.unsqueeze(0)).to(torch.float32).detach()

        aggregated_bboxes = (aggregation_mask @ bboxes) / torch.sum(aggregation_mask, -1, keepdim=True)
        aggregated_logits = (aggregation_mask @ logits) / torch.sum(aggregation_mask, -1, keepdim=True)
        
        return aggregated_bboxes, aggregated_logits
        