#!/usr/bin/env bash

PWD=$(dirname "$0")
DATA="$1"

python $PWD/../../preprocess.py --source-lang en --target-lang de \
    --trainpref $DATA/train --validpref $DATA/valid --testpref $DATA/test \
    --destdir $DATA/tokenized \
    --workers 32
