
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

from .lsq_plus import *
from ._quan_base_plus import *
from .attention_layer import *
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
from torch.overrides import has_torch_function, handle_torch_function
from .attention_layer import *
from .ms_attention_layer import *
import pdb

# sapm
# from .quant_sapm import *
from .quant_sapm_separableconv import *
from torchvision.ops import RoIAlign

def save_decoder_layer_output_as_3d_image(output_tensor, file_path="decoder_layer_output_3d_visualization.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # 移除batch维度，得到 (1800, 256) 的矩阵
    output = output_tensor.squeeze(0).detach().cpu().numpy()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(output.shape[0])  # 1800个tokens位置
    y = np.arange(output.shape[1])  # 256个特征维度
    X, Y = np.meshgrid(x, y)
    Z = output.T  # 转置为 (256, 1800) 以匹配X和Y的维度
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Visualization of Decoder Layer Output")
    ax.set_xlabel("Token Position (1800)")
    ax.set_ylabel("Feature Dimension (256)")
    ax.set_zlabel("Feature Value")

    plt.savefig(file_path, format="png")
    plt.close()
    print(f"图像已保存到: {file_path}")

class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        n_bit=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=300,
        look_forward_twice=False,
        mixed_selection=False,
        use_checkpoint=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            n_bit,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers, use_checkpoint
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            n_bit,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        decoder_layer_merge = DeformableTransformerDecoderLayerMerge(
            d_model,
            n_bit,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            decoder_layer_merge,
            num_decoder_layers,
            return_intermediate_dec,
            look_forward_twice,
            use_checkpoint,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

###########################full precision###########################
        if two_stage:
            # self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output = LinearLSQ(d_model, d_model, nbits_w=n_bit)
            self.enc_output_norm = nn.LayerNorm(d_model)
            # self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans = LinearLSQ(d_model * 2, d_model * 2, nbits_w=n_bit)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self.mixed_selection = mixed_selection
        self._reset_parameters()
####################################################################

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, QuantMultiheadAttention):
                m._reset_parameters()
            if isinstance(m, QuantMS_DAttention):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N_, 1, 1, 2
            )
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, srcs, masks, pos_embeds, query_embed=None, self_attn_mask=None):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory
            )
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
                + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                # tgt = None
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            self_attn_mask,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_bit=4,
        d_ffn=1024, 
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        # self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = QuantMS_DAttention(d_model, n_levels, n_heads, n_points, n_bit=n_bit)
        # self.self_attn = QuantMultiheadAttention(d_model, n_levels, n_heads, n_points, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        # self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear1 = LinearLSQ(d_model, d_ffn, nbits_w=n_bit)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(d_ffn, d_model)
        self.linear2 = LinearLSQ(d_ffn, d_model, nbits_w=n_bit)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                )
            else:
                output = layer(
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                )

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_bit=4,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        # self.cross_attn = QuantMS_DAttention(d_model, nhead, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.cross_attn = QuantMS_DAttention(d_model, n_levels, n_heads, n_points, n_bit=n_bit)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = QuantMultiheadAttention(d_model, n_heads, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = LinearLSQ(d_model, d_ffn, nbits_w=n_bit)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = LinearLSQ(d_ffn, d_model, nbits_w=n_bit)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoderLayerMerge(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_bit=4,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        # self.cross_attn = QuantMS_DAttention(d_model, nhead, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.cross_attn = QuantMS_DAttention(d_model, n_levels, n_heads, n_points, n_bit=n_bit)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = QuantMultiheadAttention(d_model, n_heads, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # merge attention
        self.merge_attn = QuantMultiheadAttention(d_model, n_heads, n_bit=n_bit, dropout=dropout, dist_align=False)
        self.dropout_merge = nn.Dropout(dropout)
        self.norm_merge = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = LinearLSQ(d_model, d_ffn, nbits_w=n_bit)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = LinearLSQ(d_ffn, d_model, nbits_w=n_bit)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        self_attn_mask=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        # q: Tensor, shape=[bs, num_queries, d_model]
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # merge attention
        # tgt: shape=[1, 1800, 256]
        # num_queries = tgt.shape[1]
        tgt_merged = self.merge_attn(
            tgt.transpose(0,1),
            tgt.transpose(0,1),
            tgt.transpose(0,1)
        )[0].transpose(0,1)
        tgt = tgt + self.dropout_merge(tgt_merged)
        tgt = self.norm_merge(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

# sapm
def reverse_restore_feature_maps(src, src_spatial_shapes, bs, channels):
    """
    逆向恢复特征图。

    参数:
    - src: 输入特征张量，形状为 (batch_size, num_elements, channels)
    - src_spatial_shapes: 各个特征图的空间形状，形状为 (num_levels, 2) 
                          其中每一行为 (height, width)
    - bs: batch size
    - channels: 通道数

    返回:
    - features: 恢复后的特征图列表
    """
    features = []
    start_index = 0
    for (h, w) in src_spatial_shapes:
        num_elements = h * w
        end_index = start_index + num_elements
        feature = src[:, start_index:end_index, :]  # 分割出每个尺度的特征图
        feature = feature.transpose(1, 2).reshape(bs, channels, h, w)  # 恢复原始形状
        features.append(feature)
        start_index = end_index
    return features

def plot_histogram_with_stats(tensor, save_path):
 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_flat = tensor.cpu().numpy().flatten()
    
    # 创建绘图
    plt.figure(figsize=(6, 4))
    sns.histplot(output_flat, bins=500, kde=True, color="skyblue", stat="count")

    plt.xlim((-1, 1))
    
    # 计算均值和标准差
    mean = np.mean(output_flat)
    std_dev = np.std(output_flat)
    
    # 添加均值和标准差到图中
    plt.title("quantizated output")
    plt.xlabel("values")
    plt.ylabel("count")
    plt.text(0.05, max(plt.gca().get_ylim()) * 0.8, f"mean: {mean:.5f}\nstd: {std_dev:.5f}", fontsize=10)
    
    # 去除网格线
    plt.grid(False)
    
    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存

class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        decoder_layer_merge,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        use_checkpoint=False,
    ):
        super().__init__()
        # self.layers = _get_clones(decoder_layer, num_layers)
        
        self.layers = _get_clones(decoder_layer, num_layers-1)
        self.layers.append(copy.deepcopy(decoder_layer_merge))
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice
        self.use_checkpoint = use_checkpoint
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # sapm
        self.channels = 256
        # self.q1 = 300
        # self.q2 = 1500
        # self.sapm_deformable_1 = SAPM_Deformable(self.channels, self.q1, num_scales=4)
        # self.sapm_deformable_2 = SAPM_Deformable(self.channels, self.q2, num_scales=4)

        # self.sapm_local = SAPM_Deformable(self.channels, 1, num_scales=4)
        self.sapm_local = SAPM(self.channels, 1)
        self.box_head = MLP(self.channels, self.channels, 4, 3)
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=-1)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        output = tgt 

        # sapm
        bs, _, channels = query_pos.size()     # [2, 1800, 256]
        features = reverse_restore_feature_maps(src, src_spatial_shapes, bs, channels)  # 逆向恢复 特征图 
        # Q_c0_1 = self.sapm_deformable_1(features)  # [2, 300, 256]
        # Q_c0_2 = self.sapm_deformable_2(features)  # [2, 1500, 256]
        # Q_c0 = Q_c0_1 if not self.training else torch.cat((Q_c0_1, Q_c0_2), dim=1)
        # output = Q_c0 + output
 
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    self_attn_mask,
                )
            else:
                output = layer(
                    output,
                    query_pos,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    self_attn_mask,
                )

                # sapm
                outputs_coord = self.box_head(output).sigmoid()  # [2, 300, 256] -> [2, 300, 4]
                # pooled_features = []
                # for feature,feature_shape in zip(features,src_spatial_shapes):
                #     rois = convert_to_rois(outputs_coord,feature_shape) # [2*300, 5]
                #     pooled_feature = self.roi_align(feature, rois)  # [2*300, 256, 7, 7]
                #     pooled_features.append(pooled_feature)
                # Q_c_local = self.sapm_local(pooled_features).view(bs, -1, channels) # [2*300, 1, 256] -> [2, 300, 256]

                # only finel feature
                # features[3] 1x256x13x19
                rois = convert_to_rois(outputs_coord,src_spatial_shapes[3]) # [2*300, 5]
                pooled_feature = self.roi_align(features[3], rois)  # [2*300, 256, 7, 7]
                
                Q_c_local = self.sapm_local(pooled_feature).view(bs, -1, channels)
                # if not torch.all(Q_c_local == 0.0):
                #     print("++++++++++++++++++")
                # print(torch.all(Q_c_local == 0.0))
                # print(Q_c_local)
                # if lid==5:
                #     import pdb;pdb.set_trace()
                #     plot_histogram_with_stats(output,'/data/zhangbilang/qhdetr20241018/scripts/6th_output_his_wo_local')
                #     plot_histogram_with_stats(Q_c_local,'/data/zhangbilang/qhdetr20241018/scripts/6th_local')

                output = Q_c_local + output
                # if lid==5:
                #     plot_histogram_with_stats(output,'/data/zhangbilang/qhdetr20241018/scripts/6th_output_his_with_local')
                #     import pdb;pdb.set_trace()

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    # pdb.set_trace()
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        n_bit = args.n_bit,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries_one2one + args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        look_forward_twice=args.look_forward_twice,
        use_checkpoint=args.use_checkpoint,
    )

