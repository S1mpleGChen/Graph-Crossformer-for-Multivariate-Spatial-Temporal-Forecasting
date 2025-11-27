import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil
from cross_models.attn import CrossNodeAttention

class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: Tensor with shape [B, N, D, L, C]
        Output: [B, N, D, L_new, C]  where L_new = ceil(L / win_size)
        """
        B, N, D, L, C = x.shape

        pad_num = L % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            pad = x[:, :, :, -1:, :].expand(-1, -1, -1, pad_num, -1)
            x = torch.cat([x, pad], dim=3)  # pad on segment dimension

        # 分组合并 segment
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, :, i::self.win_size, :])  # [B, N, D, L//win_size, C]
        
        x = torch.cat(seg_to_merge, dim=-1)  # [B, N, D, L//win_size, win_size*C]
        x = self.norm(x)
        x = self.linear_trans(x)  # Linear: win_size * C -> C
        return x  # [B, N, D, L_new, C]

class scale_block(nn.Module):
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout,
                 seg_num=10, factor=10, use_cross_node=True):  # 增加 use_cross_node
        super(scale_block, self).__init__()
        self.use_cross_node = use_cross_node

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList([
            TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout, use_cross_node=self.use_cross_node)
            for _ in range(depth)
        ])

        if self.use_cross_node:
            self.cross_node_attn = CrossNodeAttention(n_heads, dropout)

    def forward(self, x, topk_index=None):
        # x: [B, N, D, L, C]
        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x, topk_index=topk_index)  

        return x

class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                 in_seg_num=10, factor=10, use_cross_node=True):  # 添加 use_cross_node 参数
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout,
                                              seg_num=in_seg_num, factor=factor,
                                              use_cross_node=use_cross_node))
        for i in range(1, e_blocks):
            seg_num = ceil(in_seg_num / win_size ** i)
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,
                                                  seg_num=seg_num, factor=factor,
                                                  use_cross_node=use_cross_node))

    def forward(self, x, topk_index=None):  # 添加 topk_index 参数
        encode_x = [x]
        for block in self.encode_blocks:
            x = block(x, topk_index)
            encode_x.append(x)
        return encode_x