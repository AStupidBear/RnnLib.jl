#!/bin/bash
USE_GPU=0 python $(dirname $0)/../src/rnn.py \
    --data_path=/mnt/ST8000/rnn_regress.h5 \
    --hidden_sizes=100,100,10 \
    --layer=GRU \
    --sequence_size=500 \
    --validation_split=0.5 \
    --optimizer=AdamW \
    --lr=0.001 \
    --dropout=0 \
    --epochs 10000 \
    --l2=1e-10