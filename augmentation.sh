#!/bin/bash

# 실험1
# python "/Users/mungughyeon/Library/CloudStorage/GoogleDrive-moonstalker9010@gmail.com/내 드라이브/likelion/smartphone_defect_segmentation/src/segtool/augmentation.py" cli_augment_dataset \
#     --input_dir="/Users/mungughyeon/Library/CloudStorage/GoogleDrive-moonstalker9010@gmail.com/내 드라이브/likelion/smartphone_defect_segmentation/data/Mobile Phone Defect" \
#     --ratios="good:0.3,oil:0.3,stain:0.3"

# 실험2
python "src/segtool/augmentation.py" cli_augment_dataset \
    --input_dir="data/Mobile Phone Defect" \
    --multi_stage=True \
    --target_count=400 \
    --basic_aug_ratio=0.3