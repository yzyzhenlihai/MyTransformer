import torch

def create_padding_mask(seq_q, seq_k, pad_idx):
    """"
    创建padding mask来屏蔽K中的<pad>标记,Q不需要屏蔽,因为Q中的<pad>的部分最终计算loss时会被过滤掉
    参数:
        seq_q: 查询序列，形状为 (batch_size, len_q) 这里输入的是tokenID序列
        seq_k: 键序列，形状为 (batch_size, len_k)
        pad_idx: <pad>标记的索引ID
    返回
        pad_mask: 形状为 (batch_size, len_q, len_k) 的掩码张量，其中True表示对应位置是<pad>标记，应被屏蔽
    """
    batch_size, k_len = seq_k.size()
    q_len = seq_q.size(1)

    pad_mask = (seq_k == pad_idx).unsqueeze(1) # pad_mask.shape(batch_size, 1, k_len)

    # 广播到（N，q_len, k_len）
    return pad_mask.expand(batch_size, q_len, k_len)


def create_subsequent_mask(seq_q):
    """
    创建subsequent mask，用来屏蔽decoder self-attention中的未来位置
    参数：
        seq_q: 查询序列，形状为 (batch_size, len_q)
    返回：
        subsequent_mask: 形状为 (batch_size, len_q, len_q) 的掩码张量，其中True表示对应位置是未来位置，应被屏蔽
    """

    batch_size, q_len = seq_q.size()
    subsequent_mask = torch.triu(torch.ones((q_len, q_len),device=seq_q.device, dtype=torch.bool),
                                 diagonal=1) # (q_len, q_len)，上三角矩阵，主对角线以上为True

    return subsequent_mask.unsqueeze(0).expand(batch_size, q_len, q_len)  # (batch_size, q_len, q_len)