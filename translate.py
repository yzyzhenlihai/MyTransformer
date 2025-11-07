import torch
import os
from src.transformer.models.transformer import Transformer
try:
    # 导入与 Transformer 类在同一文件中的掩码函数
    from src.transformer.models.transformer import create_padding_mask, create_subsequent_mask
except ImportError:
    print("="*50)
    print("错误: 无法从 'src.transformer.models.transformer' 导入掩码函数。")
    print("请确保 'create_padding_mask' 和 'create_subsequent_mask' 函数")
    print("与 'Transformer' 类在同一个 .py 文件中，以便此脚本可以导入它们。")
    print("="*50)
    exit(1)

from hydra.utils import instantiate
from hydra import initialize, compose
from tokenizers import Tokenizer 

with initialize(version_base=None, config_path="./config"):
    cfg = compose(config_name="transformer.yaml")

def translate_sentence(model: Transformer, 
                         sentence: str, 
                         src_tokenizer: Tokenizer, 
                         tgt_tokenizer: Tokenizer, 
                         device, 
                         max_len: int = 100) -> str:
    """
    执行贪心解码 (Greedy Decode) 来翻译单个句子
    """

    model.eval() 

    # 1. 从 cfg 获取特殊 token 索引
    bos_idx = cfg.tokens.bos_idx
    eos_idx = cfg.tokens.eos_idx
    pad_idx = cfg.tokens.pad_idx

    # 2. 分词并转换为索引，然后添加 BOS/EOS 索引
    src_encoding = src_tokenizer.encode(sentence.lower())
    src_indices = [bos_idx] + src_encoding.ids + [eos_idx]

    # 3. 转换为 Tensor (batch_size = 1)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device) # <--- 类型为 Long

    # 4. 创建源掩码
    src_mask = create_padding_mask(src_tensor, src_tensor, pad_idx).to(device)

    # 5. 计算 Encoder 输出 (只需一次)
    with torch.no_grad():

        encoder_input = model.position_embed(model.src_embed(src_tensor)) 
        encoder_output = model.encode(encoder_input, src_mask) 


    # 6. 自回归解码 (逐个 token 预测)
    trg_indices = [bos_idx] 

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device) 

        # 7. 创建目标掩码
        tgt_pad_mask = create_padding_mask(trg_tensor, trg_tensor, pad_idx).to(device)
        tgt_subsequent_mask = create_subsequent_mask(trg_tensor).to(device)
        dec_self_mask = tgt_pad_mask | tgt_subsequent_mask
        enc_dec_mask = create_padding_mask(trg_tensor, src_tensor, pad_idx).to(device)

        # 8. 获取 Decoder 输出
        with torch.no_grad():
        
            decoder_input = model.position_embed(model.tgt_embed(trg_tensor)) 
            
            # 现在可以将 FloatTensor 送入 decode 方法
            decoder_output = model.decode(decoder_input, encoder_output, dec_self_mask, enc_dec_mask)
            
            # 使用正确的 'generator' 属性
            output = model.generator(decoder_output) # (B, Seq_Len, Vocab_Size)


        # 9. 获取最后一个 token 的预测 (贪心)
        pred_token_idx = output.argmax(2)[:, -1].item()

        trg_indices.append(pred_token_idx)

        # 10. 如果预测到 <eos> 则停止
        if pred_token_idx == eos_idx: 
            break

    # 11. 将索引转换回 Token，并跳过特殊 token
    return tgt_tokenizer.decode(trg_indices, skip_special_tokens=True)

def run_translation():

    DEVICE = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. 加载分词器
    print(f"正在加载源分词器: {cfg.data.src_tokenizer_path}")
    src_tokenizer = Tokenizer.from_file(cfg.data.src_tokenizer_path)
    
    print(f"正在加载目标分词器: {cfg.data.tgt_tokenizer_path}")
    tgt_tokenizer = Tokenizer.from_file(cfg.data.tgt_tokenizer_path)

    # 从分词器获取词表大小
    SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
    TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()

    print(f"源词表大小: {SRC_VOCAB_SIZE}, 目标词表大小: {TGT_VOCAB_SIZE}")

    # 2. 使用 Hydra 配置重新构建模型
    print("正在构建模型...")
    model = instantiate(
        cfg.model.transformer, 
        src_vocab_size=SRC_VOCAB_SIZE, 
        tgt_vocab_size=TGT_VOCAB_SIZE
    ).to(DEVICE)

    # 3. 加载训练好的模型权重
    model_path_from_log = "./checkpoints/best_model_en_to_de_10.pth" 
    print(f"正在加载模型权重: {model_path_from_log}")
    try:
        model.load_state_dict(torch.load(model_path_from_log, map_location=DEVICE))
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path_from_log}。")
        print("请确保路径正确。")
        return
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("这可能是因为模型架构与保存的权重不匹配。")
        return

    # 4. 执行翻译
    print(f"模型加载完毕。请输入一个 {cfg.data.lang_src} 句子 (或输入 'q' 退出):")

    while True:
        sentence = input(f"\n[{cfg.data.lang_src}]: ")
        if sentence.lower() == 'q':
            break

        translation = translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, DEVICE)
        print(f"[{cfg.data.lang_tgt}] : {translation}")

if __name__ == "__main__":
    run_translation()