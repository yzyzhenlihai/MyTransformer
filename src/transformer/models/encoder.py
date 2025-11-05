import torch
import torch.nn as nn
from ..components.attention import MultiHeadAttention
from ..components.feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_attn=0.1, dropout_posffn=0.1, dropout_emb=0.1):
        super(EncoderLayer, self).__init__()
        # d_model必须可以被num_heads整除
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        # 多头注意力层
        self.self_attn = MultiHeadAttention(d_k=self.d_k, d_v=self.d_v, d_model=d_model,num_heads=num_heads, dropout=dropout_attn)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_emb)

        # 位置前馈神经网络层
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout_posffn)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_emb)

    def forward(self, x, mask=None):
        """
        x.shape (batch, seq_len, d_model)
        mask.shape (batch, seq_len, seq_len)
        """
        # 多头自注意力子层
        _x = x # 缓存x用于残差连接
        x = self.self_attn(Q=x, K=x, V=x, attn_mask=mask)
        x = self.norm1(self.dropout1(x) + _x)
        
        # 位置前馈网络子层
        _x = x
        x = self.pos_ffn(x)
        x = self.norm2(self.dropout2(x) + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout_attn=0.1, dropout_posffn=0.1, dropout_emb=0.1):
        super(Encoder, self).__init__()      
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout_attn=dropout_attn, 
                         dropout_posffn=dropout_posffn, dropout_emb=dropout_emb)     
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

