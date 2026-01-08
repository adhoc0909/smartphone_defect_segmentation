#!/bin/bash

# 실험1
 python "src/segtool/augmentation.py" cli_augment_dataset \
    --input_dir="data/Mobile Phone Defect" \
    --counts="oil:100,stain:100,scratch:100"

# 실험2
# python "src/segtool/augmentation.py" cli_augment_dataset \
#     --input_dir="data/Mobile Phone Defect" \
#     --multi_stage=True \
#     --target_count=400 \
#     --basic_aug_ratio=0.3