import torch
import torch.nn as nn
from einops import rearrange

class DSW_embedding(nn.Module):
    """
    多节点输入的 DSW 切分嵌入模块。
    输入形状: (B, N, T, D)
    输出形状: (B, N, D, L_seg, d_model)
    """
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        """
        x: Tensor, shape (B, N, T, D)
        """
        B, N, T, D = x.shape
        L_seg = T // self.seg_len
        assert T % self.seg_len == 0, f"时间维度 T={T} 必须能整除 seg_len={self.seg_len}"

        # 重排：每段 segment -> (B, N, D, L_seg, seg_len)
        x = rearrange(x, 'b n (l s) d -> b n d l s', s=self.seg_len)
        self.linear = self.linear.to(x.device)

        # 对每个 seg_len 做 Linear -> d_model
        x_embed = self.linear(x)  # (B, N, D, L_seg, d_model)

        return x_embed