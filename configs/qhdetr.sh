#!/usr/bin/env bash

set -x

EXP_DIR=exps/LIEM-signed-kernel
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --epochs 25 \
    --lr_drop 10 \
    --weight_decay 0.0 \
    --num_queries_one2one 300 \
    --num_queries_one2many 1500 \
    --k_one2many 6 \
    --lambda_one2many 1.0 \
    --dropout 0.0 \
    --mixed_selection \
    --look_forward_twice \
    --quant \
    ${PY_ARGS}
