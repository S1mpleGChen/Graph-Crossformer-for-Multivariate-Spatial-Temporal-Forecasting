import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B_L, L, C = queries.shape  # B_L = B * N * D
        _, S, _ = keys.shape
        H = self.n_heads

        # Linear projection + reshape
        queries = self.query_projection(queries).view(B_L, L, H, -1)
        keys= self.key_projection(keys).view(B_L, S, H, -1)
        values= self.value_projection(values).view(B_L, S, H, -1)

        # Full attention
        out = self.inner_attention(queries, keys, values)  # [B_L, L, H, head_dim]

        # Flatten heads
        out = out.view(B_L, L, -1)
        return self.out_projection(out)

class CrossNodeAttention(nn.Module):
    def __init__(self, n_heads, dropout=0.1):
        super(CrossNodeAttention, self).__init__()
        self.cross_attn = None  # 延迟初始化
        self.n_heads = n_heads
        self.dropout_rate = dropout

        # 占位，等 forward 时根据 C 初始化
        self.norm = None
        self.mlp = None
        self.dropout = None

    def forward(self, x, topk_index):
        """
        x: Tensor [B, N, D, L, C]
        topk_index: LongTensor [B, N, K]
        """
        B, N, D, L, C = x.shape
        d_model = L * C

        # 延迟初始化 cross_attn、norm、mlp（仅初始化一次）
        if self.cross_attn is None:
            self.cross_attn = AttentionLayer(d_model, self.n_heads, dropout=self.dropout_rate).to(x.device)
            self.norm = nn.LayerNorm(C).to(x.device)  #对最后一维 C 做归一化
            self.mlp = nn.Sequential(
                nn.Linear(C, C * 2),
                nn.GELU(),
                nn.Linear(C * 2, C)
            ).to(x.device)
            self.dropout = nn.Dropout(self.dropout_rate).to(x.device)

        output = torch.zeros_like(x,device=x.device)

        for b in range(B):
            for i in range(N):
                # 当前节点 Query：[D, L, C] → [1, D, L*C]
                q = x[b, i]
                q_flat = rearrange(q, 'd l c -> 1 d (l c)')

                # Top-k 邻居 Key/Value：[K, D, L, C] → [1, K*D, L*C]
                neighbors = topk_index[b, i]
                kv = x[b, neighbors]
                kv_flat = rearrange(kv, 'k d l c -> 1 (k d) (l c)')

                # Cross Attention 输出：[1, D, L*C] → [D, L, C]
                context = self.cross_attn(q_flat, kv_flat, kv_flat)
                context = rearrange(context, '1 d (l c) -> d l c', l=L, c=C)

                # 残差 + LN + MLP
                residual = x[b, i]  # [D, L, C]
                fused = self.norm(residual + self.dropout(context))  # [D, L, C]
                fused = fused + self.dropout(self.mlp(fused))

                output[b, i] = fused

        return output

class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, top_k=3, use_cross_node=True):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.cross_node_attention = CrossNodeAttention(n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.use_cross_node = use_cross_node 

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        self.top_k = top_k

    def forward(self, x, topk_index=None):
        """
        x: Tensor of shape [B, N, D, L, C]
        topk_index: LongTensor [B, N, K]
        """

        B, N, D, L, C = x.shape

        # === 1. Time-stage TSA：每个节点的每个变量在时间维度上交互 ===
        time_in = rearrange(x, 'b n d l c -> (b n d) l c')  # [B*N*D, L, C]
        time_enc = self.time_attention(time_in, time_in, time_in)
        time_out = time_in + self.dropout(time_enc)
        time_out = self.norm1(time_out)
        time_out = time_out + self.dropout(self.MLP1(time_out))
        time_out = self.norm2(time_out)
        # 恢复为 [B, N, D, L, C]
        dim_in = rearrange(time_out, '(b n d) l c -> b n d l c', b=B, n=N, d=D)

        # === 2. Dim-stage TSA：每个节点每个时间点内的变量做 attention ===
        dim_in = rearrange(dim_in, 'b n d l c -> (b n l) d c')  # [B*N*L, D, C]
        router_expand = repeat(self.router, 'l k c -> (b n l) k c', b=B, n=N)

        dim_buffer = self.dim_sender(router_expand, dim_in, dim_in)
        dim_receive = self.dim_receiver(dim_in, dim_buffer, dim_buffer)

        dim_out = dim_in + self.dropout(dim_receive)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out))
        dim_out = self.norm4(dim_out)

        # 恢复维度为 [B, N, D, L, C]
        dim_out = rearrange(dim_out, '(b n l) d c -> b n d l c', b=B, n=N)

        # === 3. Cross-node attention：每个节点与其 top-k 邻居做融合 ===
        if self.use_cross_node and topk_index is not None:
            dim_out = self.cross_node_attention(dim_out, topk_index)

        return dim_out