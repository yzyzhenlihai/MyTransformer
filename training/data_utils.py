import os
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import matplotlib.pyplot as plt
import random
import numpy as np

def train_or_load_tokenizer(raw_dataset, lang, tokenizer_path, special_tokens_list):
    """在数据集上训练或加载一个 BPE 分词器"""
    if os.path.exists(tokenizer_path):
        print(f"加载已存在的分词器: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"为 {lang} 训练新的分词器...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=30000,  # 限制词典大小维30000
            special_tokens=special_tokens_list) # 特殊标记列表，像"[PAD]", "[UNK]", "[BOS]", "[EOS]"，预先保留这些特殊标记的ID（通常是 0, 1, 2, 3）
        
        def get_text_iterator():
            for item in raw_dataset:
                yield item['translation'][lang]

        tokenizer.train_from_iterator(get_text_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
        print(f"分词器已保存到: {tokenizer_path}")
        
    return tokenizer #返回一个分词器对象

def preprocess_fn(batch, src_tokenizer, tgt_tokenizer, lang_src, lang_tgt, bos_idx, eos_idx): 
    """
    对批次中的每个样本进行分词和添加BOS/EOS。
    """
    src_texts = [item[lang_src] for item in batch["translation"]]
    tgt_texts = [item[lang_tgt] for item in batch["translation"]]

    # 将文本列表转化为Encoding对象列表
    src_encodings = src_tokenizer.encode_batch(src_texts)
    tgt_encodings = tgt_tokenizer.encode_batch(tgt_texts)

    # 在开头和结尾添加BOS和EOSID
    batch["src_ids"] = [[bos_idx] + e.ids + [eos_idx] for e in src_encodings]  
    batch["tgt_ids"] = [[bos_idx] + e.ids + [eos_idx] for e in tgt_encodings] 
    
    return batch

class CollateFn:
    """
    在训练阶段请求下一批次时，DataLoader会在内部自动调用__call__方法,并返回结果
    批处理与填充，负责将多个独立样本组合成可以被模型直接使用的批次
    """
    def __init__(self, pad_idx_src, pad_idx_tgt):
        # 初始化填充ID
        self.pad_idx_src = pad_idx_src 
        self.pad_idx_tgt = pad_idx_tgt

    def __call__(self, batch):
        # 将src_ids和tgt_ids取出，转化为tensor
        src_ids_list = [torch.tensor(item["src_ids"]) for item in batch]
        tgt_ids_list = [torch.tensor(item["tgt_ids"]) for item in batch]

        src_padded = pad_sequence(
            src_ids_list, 
            batch_first=True, 
            padding_value=self.pad_idx_src
        ) # 接收长度不同的Tensor列表，输出(N,seq_len)，用PAD来填充短序列末尾
        tgt_padded = pad_sequence(
            tgt_ids_list, 
            batch_first=True, 
            padding_value=self.pad_idx_tgt
        )
        # 对序列进行移位
        tgt_input = tgt_padded[:, :-1] # decoder输入
        tgt_output = tgt_padded[:, 1:] # decoder下一步需要预测的

        return src_padded, tgt_input, tgt_output
    

def draw_loss_curve(train_losses, val_losses, save_path):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    #确保cuDNN使用确定性算法
    torch.backends.cudnn.deterministic = True
    # 禁用 cuDNN benchmark，benchmark 会自动寻找最快的卷积算法，但可能导致非确定性
    torch.backends.cudnn.benchmark = False
    print(f"所有随机种子已固定为 {seed}")