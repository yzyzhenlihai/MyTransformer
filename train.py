import torch
import torch.nn as nn
from src.transformer.models.transformer import Transformer
from hydra.utils import instantiate
from hydra import initialize, compose
from datasets import load_dataset
from torch.utils.data import DataLoader
from training.data_utils import train_or_load_tokenizer, preprocess_fn, CollateFn, draw_loss_curve, set_seed
from training.scheduler import NoamLR
from training.engine import train_epoch, evaluate_epoch
import os 
import matplotlib.pyplot as plt
import swanlab as wandb

# model = instantiate(cfg.model.transformer)

def train():
    with initialize(config_path="./config"):
        cfg = compose(config_name="transformer.yaml")

    if cfg.training.open_swanlab:
        wandb.login(api_key=cfg.training.swanlab_key, save=True)
        wandb.init(
            project="Transformer-Translation",
            config=cfg,
            experiment_name="Transformer-Experiment"
        )
    # 设置随机种子以确保结果可复现
    set_seed(cfg.training.seed)
    # cache_dir = os.path.join(os.getcwd(), ".cache", "huggingface", "datasets")
    # os.makedirs(cache_dir, exist_ok=True)
    raw_dataset = load_dataset(
        path=cfg.data.scripts_path,  # 使用内置的 iwslt2017 加载器
        name=cfg.data.data_name,      # 指定 iwslt2017-zh-en 这个语言对
        data_dir=cfg.data.data_dir,  # 告诉加载器去哪里查找本地文件,内部会自动拼接name来找数据 
        #cache_dir=cache_dir,
        trust_remote_code=True
        #data_dir=None
    )
    train_dataset = raw_dataset["train"]
    print(f"数据集已加载: {train_dataset}")
    print(f"第一个样本: {train_dataset[0]}")

    # 创建分词器特殊标记
    special_tokens_list = [
        cfg.tokens.pad_token,
        cfg.tokens.unk_token,
        cfg.tokens.bos_token,
        cfg.tokens.eos_token
    ]
    src_tokenizer = train_or_load_tokenizer(
        raw_dataset["train"], 
        cfg.data.lang_src, 
        cfg.data.src_tokenizer_path,
        special_tokens_list
    )
    tgt_tokenizer = train_or_load_tokenizer(
        raw_dataset["train"], 
        cfg.data.lang_tgt, 
        cfg.data.tgt_tokenizer_path,
        special_tokens_list
    )
    
    SRC_VOCAB_SIZE = src_tokenizer.get_vocab_size()
    TGT_VOCAB_SIZE = tgt_tokenizer.get_vocab_size()
    PAD_IDX_SRC = src_tokenizer.token_to_id(cfg.tokens.pad_token)
    PAD_IDX_TGT = tgt_tokenizer.token_to_id(cfg.tokens.pad_token)

    print(f"源 ({cfg.data.lang_src}) 词表大小: {SRC_VOCAB_SIZE}")
    print(f"目标 ({cfg.data.lang_tgt}) 词表大小: {TGT_VOCAB_SIZE}")
    assert PAD_IDX_TGT == cfg.tokens.pad_idx # 确保配置中的 PAD_IDX 一致

    # 对数据集进行预处理
    fn_kwargs = {
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "lang_src": cfg.data.lang_src,
        "lang_tgt": cfg.data.lang_tgt,
        "bos_idx": cfg.tokens.bos_idx, # <-- 传入 BOS ID
        "eos_idx": cfg.tokens.eos_idx  # <-- 传入 EOS ID
    }
    # 对每个分词的开头和结尾添加 BOS 和 EOS 标记
    tokenized_dataset = raw_dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=1000,
        fn_kwargs=fn_kwargs
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["translation"])

    # 创建DataLoader
    collate_fn = CollateFn(pad_idx_src=PAD_IDX_SRC, pad_idx_tgt=PAD_IDX_TGT)
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    model = instantiate(cfg.model.transformer, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE, pad_idx=PAD_IDX_TGT).to(cfg.training.device)
    print(f"模型已创建，参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_TGT, label_smoothing=0.1).to(cfg.training.device)
    # 定义优化器
    optimizer = instantiate(cfg.training.optimizer, model.parameters())
    # 定义学习率调度器
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)
    # 准备检查点目录
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{cfg.data.lang_src}_to_{cfg.data.lang_tgt}_{cfg.training.epoches}.pth")
    # 开始训练循环
    print("开始训练...")
    best_val_loss = float("inf")
    train_loss_list=[]
    val_loss_list=[]
    for epoch in range(1, cfg.training.epoches + 1):
        print(f"Epoch {epoch}/{cfg.training.epoches}")
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.training.device
        )
        val_loader_loss = evaluate_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=cfg.training.device
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loader_loss)
        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loader_loss:.4f}")
        if cfg.training.open_swanlab:
            wandb.log({
                "Train Loss": train_loss,
                "Validation Loss": val_loader_loss,
                "Epoch": epoch
            })
        if val_loader_loss < best_val_loss:
            best_val_loss = val_loader_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"新最佳模型已保存到 {best_model_path}")
    # 绘制并保存损失曲线
    figure_path = os.path.join(cfg.training.results_dir, f"loss_curve_{cfg.data.lang_src}_to_{cfg.data.lang_tgt:}.png")
    draw_loss_curve(train_loss_list, val_loss_list, figure_path)
    print("训练完成...")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型保存在: {best_model_path}")
    print(f"Loss曲线保存在: {figure_path}")

if __name__ == "__main__":
    train()
