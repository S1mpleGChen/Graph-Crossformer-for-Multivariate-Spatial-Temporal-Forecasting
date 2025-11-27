import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import ceil

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.cross_embed import DSW_embedding

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                 factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, baseline=False, device=torch.device('cuda:0'),
                 use_cross_node=True):
        super(Crossformer, self).__init__()

        self.data_dim = data_dim  # 每个节点的特征维度
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.device = device
        self.baseline = baseline

        # padding 处理
        self.pad_in_len = ceil(in_len / seg_len) * seg_len
        self.pad_out_len = ceil(out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, 1, data_dim, self.pad_in_len // seg_len, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(
            e_blocks=e_layers,
            win_size=win_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=self.pad_in_len // seg_len,
            factor=factor,
            use_cross_node=use_cross_node
        )

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, self.pad_out_len // seg_len, d_model))
        self.decoder = Decoder(
            seg_len=seg_len,
            d_layers=e_layers + 1,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            out_seg_num=self.pad_out_len // seg_len,
            factor=factor
        )

    def forward(self, x_seq, topk_index=None):
        """
        x_seq: [B, N, T, D]  -> 输入为多个节点的多变量时序数据
        topk_index: [B, N, K] 每个样本每个节点的 K 个邻居索引
        """
        B, N, T, D = x_seq.shape
        if self.in_len_add > 0:
            pad = x_seq[:, :, :1, :].expand(-1, -1, self.in_len_add, -1)
            x_seq = torch.cat([pad, x_seq], dim=2)  # 在时间维度前补齐
        assert x_seq.shape[2] == self.pad_in_len

        if self.baseline:
            base = x_seq.mean(dim=2, keepdim=True)  # [B, N, 1, D]
        else:
            base = 0

        # embedding
        x_embed = self.enc_value_embedding(x_seq)  # [B, N, D, L, C]
        x_embed = x_embed + self.enc_pos_embedding  # broadcasting: [1, 1, D, L, C]
        x_embed = self.pre_norm(x_embed)

        # encoder
        enc_out = self.encoder(x_embed, topk_index=topk_index)
        for i, layer_out in enumerate(enc_out):
            print(f"[Encoder Output] Layer {i}: shape = {layer_out.shape}")

        # decoder input
        dec_in = repeat(self.dec_pos_embedding, '1 d l c -> b n d l c', b=B, n=N)

        # decoder
        dec_out = self.decoder(dec_in, enc_out)  # [B, N, out_len, D]

        # 最终输出为每个节点的目标变量（如 power）预测：只取 D=0
        return base + dec_out[:, :, :self.out_len, 0:1]  # shape: [B, N, out_len, 1]