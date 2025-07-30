#!/bin/bash

python test_3d_ovs.py \
    --data_path "data/3d_ovs_mask/test" \
    --eval_index "assets/evaluation_index_3d_ovs.json" \
    --model_path "checkpoints/pretrained_models/checkpoint-final.pth" \
    --resolution 256 \
    --output_path "outputs/3d_ovs/"
