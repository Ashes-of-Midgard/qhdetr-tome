import torch
from torch import nn
import torch.nn.functional as F

# from .lsq_plus import *
# from ._quan_base_plus import truncation
from util import box_ops


def symmetric_cross_entropy(p, q, epsilon=1e-8):
    q = q + epsilon
    p = p + epsilon
    ce_pq = (-p * q.log()).sum(dim=-1)
    ce_qp = (-q * p.log()).sum(dim=-1)
    sce = ce_pq + ce_qp
    return sce


class OutAggregate(nn.Module):
    def __init__(self, num_classes, t_b=0.9, t_c=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.t_b = t_b
        self.t_c = t_c

    def forward(self, bboxes, logits):
        prob = logits.sigmoid()
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

            sce_matrix = symmetric_cross_entropy(prob.unsqueeze(2), prob.unsqueeze(1))
            print(f"sce_matrix:\n{sce_matrix}")
            print(sce_matrix.shape)
            kl_div_matrix_lt_threshold = sce_matrix < self.t_c
            
            aggregation_mask = iou_matrix_gt_threshold & kl_div_matrix_lt_threshold
            aggregation_mask = aggregation_mask | aggregation_mask.T # to ensure mask is symmetric

            # calculate the transitive closure 
            adj = aggregation_mask.to(torch.float32)
            t = 0
            while t < n_q:
                new_adj = ((adj + torch.matmul(adj,adj)) > 1e-6).to(torch.float32)
                if torch.all(new_adj==adj):
                    break
                adj = new_adj
                t += 1
            aggregation_mask = torch.unique(adj, dim=1)
            print(f"aggregation_mask:\n{aggregation_mask}")
            print(aggregation_mask.shape)
        
        aggregated_bboxes = (aggregation_mask @ bboxes) / torch.sum(aggregation_mask, -1, keepdim=True)
        aggregated_prob = (aggregation_mask @ prob) / torch.sum(aggregation_mask, -1, keepdim=True)
        aggregated_logits = torch.special.logit(aggregated_prob, eps=1e-6)

        return aggregated_bboxes, aggregated_logits
        
