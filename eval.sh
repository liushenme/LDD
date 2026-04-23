#!/bin/bash

name="rgb_c2d_diff_fusion2poolnnchw_r18_audio_fusionsacah4f512d0"
model="Deepfakecla_rgb_c2d_difffusion_4d_audiofusion"

date

CUDA_VISIBLE_DEVICES=0 python -u evaluate_cla_ffv.py \
    --gpus 1 \
    --config ./config/default.toml \
    --batch_size 1 \
    --name $name \
    --model $model \
    --data_root XXXX\
    --checkpoint exp/$name/checkpoints/XXXX


