import torch
import torch.nn as nn
import numpy as np
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 模型输入总维度，词嵌入嵌入的维度
        self.d_k = d_k # 每个头的Q和K维度
        self.d_v = d_v # 每个头的V的维度   通常d_k=d_v = d_model/num_heads
        self.num_heads = num_heads # 头的数量
        self.dropout = nn.Dropout(dropout)
        
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model) #输出投影层，接受所有”“头”的输出并将其映射回d_model维度

        # 对权重进行显示初始化，有助于训练开始时保持梯度稳定
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask):
        """
        Q, K, V: (N, seq_len, d_model)
        attenn_mask: (N, q_len, k_len) 用于屏蔽掉某些位置，例如padding或decoder中的未来位置
        """
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # 多头拆分
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)  # -1被自动推断为seq_len,交换维度后得到 Q.shape(N,num_heads,seq_len,d_k)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)  # K.shape(N,num_heads,seq_len,d_k)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)  # V.shaoe(N,num_heads,seq_len,d_v)
        
        # 预处理注意力掩码
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)    # attn_mask.unsqueeze(1): 形状变为 (N, 1, q_len, k_len) => repeat(1, num_heads, 1, 1): 形状变为 (N, num_heads, q_len, k_len)。
            attn_mask = attn_mask.bool()

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4) # 将被mask=true的位置的分数设为一个非常小的值，确保softmax后这些位置的权重接近于零
        attns = torch.softmax(scores, dim=-1)        # attention weights
        attns = self.dropout(attns)  # 一种正则化手段

        # calculate output
        output = torch.matmul(attns, V) # attns.shape: (N, num_heads, q_len, k_len)  V.shape: (N, num_heads, k_len, d_v) output.shape: (N, num_heads, q_len, d_v)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads) # output.shape: (N, q_len, num_heads, d_v) => output.shape: (N, q_len, d_v * num_heads)
        output = self.W_out(output)   # output.shape: (N, q_len, d_model)

        return output