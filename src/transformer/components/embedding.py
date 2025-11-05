import torch 
import torch.nn as nn

class PositionEncoding(nn.Module):
    """基于sin/cos实现的位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # position.shape (max_len, 1)
        # 计算公式的分母部分
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-torch.log(torch.tensor(10000.0))/d_model)) # div_term.shape (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # position * div_term 广播机制，shape (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe.shape (1, max_len, d_model) 增加一个维度作为batch维度

        # 将pe注册为模型的state，而不是param
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x.shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)] # pe.shape(1, seq_len, d_model) 

        return self.dropout(x)
    
class TokenEmbedding(nn.Module):
    """
    标准的词嵌入层
    """
    def __init__(self, vocab_size, d_model, pad_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def forward(self, x):
        """
        x.shape (batch, seq_len) 输入的是序列的tokenID
        """
        # 乘上sqrt(d_model)进行缩放，避免词向量太小被位置编码淹没的问题，导致训练初期难以学习到词的语义
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) 