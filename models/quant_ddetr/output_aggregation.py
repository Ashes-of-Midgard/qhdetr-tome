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
        # print(batch_idx, proposal_indices_row, proposal_indices_column, kl_divs)
        result_matrix[batch_idx, proposal_indices_row, proposal_indices_column] = kl_divs.mean(dim=-1)

    return result_matrix


class OutAggregate(nn.Module):
    def __init__(self, num_classes, t_b=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.t_b = t_b

    def forward(self, bboxes, logits):
        with torch.no_grad():
            n_q = bboxes.shape[1]
            iou_matrix = []
            for i in range(len(bboxes)):
                iou_matrix.append(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(bboxes[i]),
                    box_ops.box_cxcywh_to_xyxy(bboxes[i])
                ))
            iou_matrix = torch.stack(iou_matrix)
            iou_matrix_gt_threshold = iou_matrix > self.t_b
            upper_triangle_mask = (torch.arange(n_q).unsqueeze(0) > torch.arange(n_q).unsqueeze(1)).unsqueeze(0).to(iou_matrix_gt_threshold.device)
            aggregation_mask = upper_triangle_mask * iou_matrix_gt_threshold
            for i in range(n_q):
                aggregation_mask = (~aggregation_mask[:, i, :].unsqueeze(2)) * aggregation_mask
            eye = torch.eye(n_q, dtype=torch.bool, device=aggregation_mask.device).unsqueeze(0)
            aggregation_mask = aggregation_mask + eye
        
        aggregated_bboxes = (aggregation_mask @ bboxes) / torch.sum(aggregation_mask, -1, keepdim=True)
        return aggregated_bboxes, logits
        