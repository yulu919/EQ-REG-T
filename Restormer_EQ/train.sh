#!/usr/bin/env bash

CONFIG=$1

### for restormer
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt $CONFIG --launcher pytorch

### for restormer-reg
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train_loss.py -opt $CONFIG --launcher pytorch