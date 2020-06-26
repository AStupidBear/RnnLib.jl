#!/bin/bash
dir=$(dirname $0)
USE_GPU=0 python $dir/../src/rnn.py \
    --data_path=$HOME/job/rnn_regress.h5 \
    --hidden_sizes=100,100,10 \
    --layer=GRU \
    --sequence_size=500 \
    --validation_split=0.5 \
    --optimizer=AdamW \
    --lr=0.001 \
    --dropout=0 \
    --epochs 10000 \
    --l2=1e-10