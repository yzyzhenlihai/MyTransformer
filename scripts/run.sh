mkdir -p ./logs
nohup python train.py > ./logs/train_without_encoder.log 2>&1 & 