#!/bin/bash


name="rgb_c2d_diff_fusion2poolnnchw_r18_audio_fusionsacah4f512d0"
model="Deepfakecla_rgb_c2d_difffusion_4d_audiofusion"

date

CUDA_VISIBLE_DEVICES=0,1 python -u train_cla_ffv.py --config ./config/default.toml \
    --data_root XXXXXX \
    --model $model \
    --batch_size 4 \
    --num_workers 8 \
    --max_epochs 20 \
    --gpus 2 \
    --precision 32 \
    --exp_dir exp/$name

