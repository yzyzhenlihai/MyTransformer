### 文件目录结构
```
./MyTransformer/
├── config                                      （运行配置）
├──data                                         （数据集压缩包）
├── results                                     （训练结果图）
├── scripts                                     （一键运行脚本）
├── src
│   ├── __init_.py
│   └── transformer
│       ├── components
│       │   ├── attention.py                    （多头注意力实现）
│       │   ├── embedding.py                    （位置编码及词嵌入层实现）
│       │   ├── feed_forward.py                 （位置前馈神经网络实现）
│       │   ├── __init__.py
│       ├── models
│       │   ├── decoder.py                      （编码器实现）
│       │   ├── encoder.py                      （解码器实现）
│       │   └── transformer.py                  （Transformer顶层搭建）
│       └── utils
│           ├── __init__.py
│           ├── masking.py                      （填充掩码及后续掩码实现）
├── training
│   ├── data_utils.py                           （训练时的工具函数）
│   ├── engine.py                               （训练及评估方法）
│   ├── __init__.py
│   └── scheduler.py                            （学习率调度器实现）
└── train.py
```

### 运行方式
1. 安装依赖
   ```bash
   pip install -r ./requirements.txt
   ```
2. 解压缩 `data` 文件夹下的文件压缩包
   ```bash
   cd ./data/iwslt2017
   tar -zxvf ./en-de.zip
   ```
3. 修改 `./config/transformer.yaml` 配置文件
    ```yaml
    # 配置自己的swanlab的api key
    #也可选择将open_swanlab: false
    open_swanlab: true
    wanlab_key: "ccVCViGdYDi4LOGAy6FBp"
    ```
4. `cd` 到 `./MyTransformer`目录下一键运行脚本
    ```bash
    ./scripts/run.sh
    ```  

### 复现要求
   - **GPU：** A4000
   - **随机数种子：** 42