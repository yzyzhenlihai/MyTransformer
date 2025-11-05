import torch
import torch.nn as nn 

class PositionwiseFeedForward(nn.Module):
    """实现位置前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x.shape(batch, seq_len, d_model)
        """
        x = self.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x
