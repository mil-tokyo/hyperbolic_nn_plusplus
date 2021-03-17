#!/bin/bash

PWD=$(dirname "$0")
N_GPUS="$1"
GPU_IDS="$2"
SCRATCH="$3"
model_dim="$6" #16
dropout="$7"
n_warmup="$8" #4000
max_iters="$9" #100000

model_type=p_fconv_wmt_$model_dim
data_dir="$4"
save_dir="$5"/$model_type
log_dir=$save_dir/log

if [ $SCRATCH = "-s" ]; then # scratch
    echo "[Training from scratch (previous checkpoint will be deleted)]"
    if [ -e $save_dir ]; then
        rm -rf $save_dir
    fi
elif [ $SCRATCH = "-r" ]; then # resume
    echo "[Resuming the last checkpoint if it exists]"
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS python $PWD/../../train.py $data_dir \
    --task translation --source-lang en --target-lang de --tokenizer moses --bpe subword_nmt \
    --save-dir $save_dir --tensorboard-logdir $log_dir --log-interval 20 \
    --arch $model_type \
    --dropout=$dropout \
    --max-tokens $((10000 / $N_GPUS)) --required-batch-size-multiple 1 \
    --num-workers $((8 * $N_GPUS)) \
    --max-source-positions 1024 --max-target-positions 1024 \
    --optimizer riemannian_adam --lr `echo "1/sqrt("$model_dim"*"$n_warmup")"|bc -l` \
    --adam-betas "(0.9, 0.98)" --adam-eps "1e-9" \
    --max-update $max_iters \
    --lr-scheduler inverse_sqrt --warmup-updates $n_warmup \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-last-epochs 10 --keep-interval-updates 10 --save-interval-updates 5000 --no-epoch-checkpoints \
    --ddp-backend=no_c10d \



