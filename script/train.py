from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from segtool.config import TrainConfig, DataConfig
from segtool.data import make_loaders
from segtool.models_factory import build_model
from segtool.losses import BCEDiceLoss
from segtool.engine import train_one_epoch, validate
from segtool.utils import set_seed, get_device, ensure_dir, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, default='/content')
    p.add_argument("--out_dir", type=str, default='./output')
    p.add_argument("--run_name", type=str, default=None)
    
    p.add_argument("--img_h", type=int, default=144)
    p.add_argument("--img_w", type=int, default=256)  # !!!이미지 사이즈를 이렇게 정한 근거를 들어야 함
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    # hyper parameters
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0) # !!!overfitting 날 경우 튜닝
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5) # 확률 맵 logit에 대해서 이진 마스크로 바꿀 때 쓰는 하이퍼파라미터
    p.add_argument("--use_early_stopping", action="store_true", help="Enable early stopping") # early stopping 사용 여부
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience") # early stopping 값

    p.add_argument("--model", type=str, default='unet')
    p.add_argument("--base_channels", type=int, default=32) 
    p.add_argument("--bce_weight", type=float, default=0.5)

     # wandb
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="smartphone_defect_segmentation")
    p.add_argument("--wandb_entity", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data=DataConfig(
            base_path=Path(args.base_path),
            img_size_h=args.img_h,
            img_size_w=args.img_w,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        ),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        threshold=args.threshold,
        out_dir=Path(args.out_dir),
        run_name=args.run_name or f"{args.model}_h{args.img_h}_w{args.img_w}_bs{args.batch_size}_lr{args.lr}",
        model_name=args.model,
        base_channels=args.base_channels,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    
    set_seed(cfg.data.seed)
    device = get_device()
    print("Device:", device)

    # wandb
    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config={
                "model": cfg.model_name,
                "img_h": cfg.data.img_size_h,
                "img_w": cfg.data.img_size_w,
                "train_ratio": cfg.data.train_ratio,
                "test_ratio": cfg.data.test_ratio,
                "seed": cfg.data.seed,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "threshold": cfg.threshold,
                "base_channels": cfg.base_channels,
                "loss": "BCEDice",
                "bce_weight": args.bce_weight,
                # "use_early_stopping": args.use_early_stopping,
                "patience": args.patience,
            },
        )

    train_loader, val_loader, test_loader = make_loaders(
    # base_path=cfg.data.base_path / 'data/',
    base_path=cfg.data.base_path,
    img_size_hw=(cfg.data.img_size_h, cfg.data.img_size_w),
    train_ratio=cfg.data.train_ratio,
    test_ratio=cfg.data.test_ratio,
    seed=cfg.data.seed,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    )

    model = build_model(cfg.model_name, base_channels=cfg.base_channels).to(device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = cfg.out_dir / cfg.run_name
    ensure_dir(run_dir)
    print("Run dir:", run_dir)

    best_val = -1.0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, criterion, device, threshold=cfg.threshold)
        va = validate(model, val_loader, criterion, device, threshold=cfg.threshold)

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train loss {tr.loss:.4f} | dice(all) {tr.metrics_all.dice:.4f} iou {tr.metrics_all.iou:.4f} "
            f"| val loss {va.loss:.4f} | dice(all) {va.metrics_all.dice:.4f} iou {va.metrics_all.iou:.4f} "
            f"| val dice(defect) {va.metrics_defect_only.dice:.4f} iou(defect) {va.metrics_defect_only.iou:.4f}"
        )

        if cfg.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": tr.loss,
                "train/dice_all": tr.metrics_all.dice,
                "train/iou_all": tr.metrics_all.iou,
                "val/loss": va.loss,
                "val/dice_all": va.metrics_all.dice,
                "val/iou_all": va.metrics_all.iou,
                "val/dice_defect": va.metrics_defect_only.dice,
                "val/iou_defect": va.metrics_defect_only.iou,
                "val/precision_defect": va.metrics_defect_only.precision,
                "val/recall_defect": va.metrics_defect_only.recall,
            })

        if epoch % cfg.save_every == 0:
            save_checkpoint(run_dir / "last.pt", model, optim, epoch, extra={"run_name": cfg.run_name})

        if va.metrics_defect_only.dice > best_val:
            best_val = va.metrics_defect_only.dice
            patience_counter = 0  # Reset patience counter
            save_checkpoint(run_dir / "best.pt", model, optim, epoch, extra={"best_val_defect_dice": best_val})
            print(f"  -> saved best.pt (defect dice {best_val:.4f})")
            if cfg.use_wandb:
                import wandb
                wandb.run.summary["best_val_defect_dice"] = best_val
        else:
            patience_counter += 1
            
        # Early stopping check
        if args.use_early_stopping and patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (patience {args.patience})")
            break

    if cfg.use_wandb and wandb_run is not None:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
