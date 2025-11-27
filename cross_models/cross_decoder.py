import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer


class DecoderLayer(nn.Module):
    '''
    图结构 Crossformer 的 DecoderLayer：
    - 每个节点做 TSA
    - 与 Encoder 的对应层进行 Cross Attention 融合
    - 最后做 MLP 和预测（仅对变量0如 Power）
    '''

    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num=10, factor=10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, d_ff, dropout, use_cross_node=False )
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x: Decoder 输入 [B, N, D, L, C]
        cross: 对应 Encoder 层输出 [B, N, D_enc, L_enc, C]
        '''
        device = x.device
        self.to(device)
        B, N, D, L, C = x.shape

        # 1. TSA (节点内部)
        
        x = self.self_attention(x)
        print(f"[DecoderLayer] x.shape before cross attention: {x.shape}")
        

        # 2. Cross Attention (对每个变量序列)
        q = rearrange(x, 'b n d l c -> (b n d) l c')
        print(f"[DecoderLayer] q.shape after rearrange: {q.shape}")
        k = rearrange(cross, 'b n d l c -> (b n d) l c')
        print(k.shape)
        v = k
        tmp = self.cross_attention(q, k, v)
        tmp = rearrange(tmp, '(b n d) l c -> b n d l c', b=B, n=N, d=D)
        x = x + self.dropout(tmp)

        # 3. FFN
        y = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + self.dropout(y))

        # 4. 仅预测变量0（如 power），形状 [B, N, L, seg_len]
        layer_predict = self.linear_pred(dec_output)  # [B, N, D, L, seg_len]
        layer_predict = layer_predict[:, :, 0, :, :]  # 只用变量0预测

        return dec_output, layer_predict


class Decoder(nn.Module):
    '''
    Crossformer Decoder：多层解码器，每层都做一次预测，最终加权融合。
    '''

    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,
                 router=False, out_seg_num=10, factor=10):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList([
            DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor)
            for _ in range(d_layers)
        ])

    def forward(self, x, cross):
        '''
        x: Decoder 输入 [B, N, D, L, C]
        cross: List，每层 Encoder 输出 [B, N, D_enc, L_enc, C]
        return: final prediction [B, N, L * seg_len, 1]
        '''
        final_predict = None
        for i, layer in enumerate(self.decode_layers):
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict

        # 还原预测维度 [B, N, L * seg_len, 1]
        final_predict = rearrange(final_predict, 'b n l s -> b n (l s) 1')
        return final_predict

