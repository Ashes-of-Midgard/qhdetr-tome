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
    def __init__(self, num_classes, t_b=0.9, t_c=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.t_b = t_b
        self.t_c = t_c

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

            kl_div_matrix = torch.kl_div(prob.unsqueeze(-1), prob)
            print(f"kl_div_matrix:\n{kl_div_matrix}")
            print(kl_div_matrix.shape)
            kl_div_matrix_lt_threshold = kl_div_matrix < self.t_c
            
            aggregation_mask = iou_matrix_gt_threshold & kl_div_matrix_lt_threshold
            aggregation_mask = aggregation_mask | aggregation_mask.T # to ensure mask is symmetric

            # calculate the transitive closure 
            adj = aggregation_mask.astype(torch.uint8)
            t = 0
            while t < n_q:
                new_adj = ((adj + adj @ adj) > 0).astype(torch.uint8)
                if torch.all(new_adj==adj):
                    break
                adj = new_adj
                t += 1
            aggregation_mask = torch.unique(adj, dim=1).astype(torch.float32)
            print(f"aggregation_mask:\n{aggregation_mask}")
            print(aggregation_mask.shape)
        
        aggregated_bboxes = (aggregation_mask @ bboxes) / torch.sum(aggregation_mask, -1, keepdim=True)
        prob = logits.sigmoid()
        aggregated_prob = (aggregation_mask @ prob) / torch.sum(aggregation_mask, -1, keepdim=True)
        aggregated_logits = torch.special.logit(aggregated_prob, eps=1e-6)

        return aggregated_bboxes, aggregated_logits
        