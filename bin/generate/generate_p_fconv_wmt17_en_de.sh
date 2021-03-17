#!/bin/bash

PWD=$(dirname "$0")
GPU_IDS="$1"

model_dim="$4" #16

model_type=p_fconv_wmt_$model_dim
data_dir="$2"
check_root="$3"/$model_type/checkpoint

CUDA_VISIBLE_DEVICES=$GPU_IDS python $PWD/../../generate.py $data_dir \
    --task translation --source-lang en --target-lang de \
    --path "$check_root"_best.pt \
    --beam 5 --batch-size 128 --remove-bpe
