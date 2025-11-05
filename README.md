my_transformer/
├── src/
│   └── transformer/
│       ├── __init__.py           # 包的入口，导出主要模型类
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── transformer.py    # [1] 完整 Transformer 模型组装
│       │   ├── encoder.py        # [2] Encoder 和 EncoderLayer
│       │   └── decoder.py        # [3] Decoder 和 DecoderLayer
│       │
│       ├── components/
│       │   ├── __init__.py
│       │   ├── attention.py      # [4] MultiHeadAttention 和 ScaledDotProductAttention
│       │   ├── feed_forward.py   # [5] PositionwiseFeedForward (FFN)
│       │   ├── embedding.py      # [6] TokenEmbedding 和 PositionalEncoding
│       │   └── sublayers.py      # [7] 残差连接 + 层归一化 (SublayerConnection)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── masking.py        # [8]  padding_mask 和 subsequent_mask 的创建函数
│           └── initialization.py # [9] (可选) 权重初始化函数
│
├── train.py                    # (项目根目录) 训练脚本
├── config.py                   # (项目根目录) 存放超参数 (d_model, n_heads, n_layers...)
└── README.md                   # (项目根目录) 你的项目说明文档