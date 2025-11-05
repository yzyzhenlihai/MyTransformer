import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from ..components.embedding import TokenEmbedding, PositionEncoding
from ..utils.masking import create_padding_mask, create_subsequent_mask

class Transformer(nn.Module):
    """
    实现Transformer模型
    参数:
    - src_vocab_size: 源语言词汇表大小
    - tgt_vocab_size: 目标语言词汇表大小
    - d_model: 词向量维度
    - num_heads: 多头注意力机制中的头数
    - d_ff: 位置前馈神经网络的隐藏层维度
    - num_layers: 编码器和解码器的层数
    - pad_idx: 用于填充的token ID
    - dropout_attn: 注意力机制中的dropout概率
    - dropout_posffn: 位置前馈神经网络中的dropout概率
    - dropout_emb: 词嵌入和位置编码中的dropout概率
    - max_len: 位置编码的最大长度
    """
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 d_ff,
                 num_layers,
                 pad_idx,
                 dropout_attn=0.1,
                 dropout_posffn=0.1,
                 dropout_emb=0.1,
                 max_len=5000):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        # 构建嵌入层
        self.src_embed = TokenEmbedding(vocab_size=src_vocab_size, d_model=d_model, pad_idx=pad_idx)
        self.tgt_embed = TokenEmbedding(vocab_size=tgt_vocab_size, d_model=d_model, pad_idx=pad_idx)
        self.position_embed = PositionEncoding(d_model=d_model, max_len=max_len, dropout=dropout_emb)

        # 构建编码器和解码器
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers,
                               dropout_attn=dropout_attn, dropout_posffn=dropout_posffn, dropout_emb=dropout_emb)
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, 
                               dropout_attn=dropout_attn, dropout_posffn=dropout_posffn, dropout_emb=dropout_emb)
        
        # 输出层，将解码器输出向量维度映射到目标词汇表大小
        self.generator = nn.Linear(d_model, tgt_vocab_size)


    def create_masks(self, src, tgt):
        # 创建Encoder自注意掩码，屏蔽padding
        src_mask = create_padding_mask(src, src, self.pad_idx)

        # 创建Decoder自注意力掩码，屏蔽padding和未来信息
        tgt_pad_mask = create_padding_mask(tgt, tgt, self.pad_idx)
        tgt_subsequent_mask = create_subsequent_mask(tgt)
        dec_self_mask = tgt_pad_mask | tgt_subsequent_mask

        # Encoder-Decoder注意力掩码，屏蔽Encoder输出的padding
        enc_dec_mask = create_padding_mask(tgt, src, self.pad_idx)

        return src_mask, dec_self_mask, enc_dec_mask


    def forward(self, src, tgt):

        """
        参数：
            src：源序列(N, src_len) 输入的是tokenID序列
            tgt：目标序列(N, tgt_len) 输入的是tokenID序列
        """
        # 创建掩码
        src_mask, dec_self_mask, enc_dec_mask = self.create_masks(src, tgt)

        # 对序列进行嵌入和位置编码
        src_emb = self.position_embed(self.src_embed(src))
        tgt_emb = self.position_embed(self.tgt_embed(tgt))

        # 编码器前向传播
        encoder_output = self.encoder(src_emb, src_mask)

        # 解码器向前传播
        decoder_output = self.decoder(tgt_emb, encoder_output, dec_self_mask, enc_dec_mask)

        # 生成最终输出
        output = self.generator(decoder_output)

        return output
    
    """用于推理时的编码和解码方法"""
    def encode(self, src,src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, dec_self_attn_mask, enc_dec_attn_mask):
        return self.decoder(tgt, encoder_output, dec_self_attn_mask, enc_dec_attn_mask)