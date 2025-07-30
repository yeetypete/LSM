#!/bin/bash

python demo.py \
    --file_list datasets/3d_ovs/bed/context/*.jpg \
    --model_path "checkpoints/pretrained_models/checkpoint-final.pth" \
    --output_path "outputs/3d_ovs/" \
    --resolution "256"
