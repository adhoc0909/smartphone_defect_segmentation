# Mobile Phone Defect Segmentation (U-Net baseline + W&B)

현업형 포트폴리오를 목표로, **재현 가능한 학습/평가/추론 파이프라인** 형태로 모듈화한 segmentation 베이스라인입니다.
- 기본 입력: 256x144 (sanity baseline)
- 실험 관리: **Weights & Biases (wandb)** (옵션)

## Dataset layout (expected)
```
/content/
  scratch/            # defect images
  oil/
  stain/
  good/               # normal images
  ground_truth_1/      # masks for scratch/stain
  ground_truth_2/      # masks for oil
```

## Install
```bash
pip install -r requirements.txt
wandb login
```

## Train (no wandb)
```bash
PYTHONPATH=src python scripts/train.py --base_path /content --epochs 10
```

## Train (with wandb)
```bash
PYTHONPATH=src python scripts/train.py --base_path /content --epochs 10 --use_wandb --wandb_project mobile-defect-seg
```

Outputs:
- `outputs/<run_name>/last.pt`
- `outputs/<run_name>/best.pt` (defect-only Dice 기준)

## Evaluate
```bash
PYTHONPATH=src python scripts/eval.py --base_path /content --ckpt outputs/<run_name>/best.pt
```

## Inference (single image)
```bash
PYTHONPATH=src python scripts/infer.py --image /content/scratch/Scr_0001.jpg --ckpt outputs/<run_name>/best.pt --out pred.png
```

## Notes
- 모델 확장은 `src/segtool/models_factory.py`에 추가하세요.
- 다음 단계로 patch 학습 + sliding-window inference로 확장 추천.
