from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

from segtool.config import DataConfig
from segtool.data import make_loaders
from segtool.models_factory import build_model
from segtool.model import FCN
from segtool.losses import BCEDiceLoss
from segtool.engine import validate
from segtool.utils import get_device, load_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--base_path", type=str, required=True, help="Path to dataset root")
    p.add_argument("--model", type=str, default="fcn", choices=["fcn", "unet"])
    p.add_argument("--base_channels", type=int, default=32)
    
    p.add_argument("--img_h", type=int, default=144)
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    
    return p.parse_args()

def evaluate_model(model, test_loader, criterion, device, threshold: float = 0.5):
    """Evaluate model performance on test set"""
    print("Starting Test Evaluation...")
    print("-" * 60)
    
    test_result = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=threshold
    )
    
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_result.loss:.4f}")
    print(f"Test Dice (All): {test_result.metrics_all.dice:.4f}")
    print(f"Test IoU (All): {test_result.metrics_all.iou:.4f}")
    print(f"Test Precision (All): {test_result.metrics_all.precision:.4f}")
    print(f"Test Recall (All): {test_result.metrics_all.recall:.4f}")
    print()
    print("DEFECT-ONLY METRICS")
    print("-" * 30)
    print(f"Test Dice (Defects): {test_result.metrics_defect_only.dice:.4f}")
    print(f"Test IoU (Defects): {test_result.metrics_defect_only.iou:.4f}")
    print(f"Test Precision (Defects): {test_result.metrics_defect_only.precision:.4f}")
    print(f"Test Recall (Defects): {test_result.metrics_defect_only.recall:.4f}")
    
    print()
    print("PERFORMANCE SUMMARY")
    print("-" * 30)
    defect_dice = test_result.metrics_defect_only.dice
    if defect_dice > 0.8:
        print("Excellent performance! (Dice > 0.8)")
    elif defect_dice > 0.6:
        print("Good performance! (Dice > 0.6)")
    elif defect_dice > 0.4:
        print("Moderate performance (Dice > 0.4)")
    else:
        print("Poor performance (Dice < 0.4)")
    
    return test_result

def load_model_from_checkpoint(checkpoint_path: str, model_type: str = "fcn", 
                              base_channels: int = 32, device=None):
    """Load model from checkpoint for jupyter usage"""
    if device is None:
        device = get_device()
    
    if model_type == "fcn":
        model = FCN(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    else:
        model = build_model(model_type, base_channels=base_channels).to(device)
    
    checkpoint = load_checkpoint(Path(checkpoint_path), model, map_location=str(device))
    print(f"Loaded model from epoch: {checkpoint['epoch']}")
    if 'best_val_defect_dice' in checkpoint:
        print(f"Best validation defect dice: {checkpoint['best_val_defect_dice']:.4f}")
    
    return model, checkpoint

def quick_evaluate(checkpoint_path: str, test_loader, device=None, model_type: str = "fcn"):
    """Quick evaluation function for jupyter notebooks"""
    if device is None:
        device = get_device()
    
    model, _ = load_model_from_checkpoint(checkpoint_path, model_type, device=device)
    criterion = BCEDiceLoss()
    return evaluate_model(model, test_loader, criterion, device)

def main():
    args = parse_args()
    device = get_device()
    print("Device:", device)
    
    data_config = DataConfig(
        base_path=Path(args.base_path),
        img_size_h=args.img_h,
        img_size_w=args.img_w,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    _, _, test_loader = make_loaders(
        base_path=data_config.base_path,
        img_size_hw=(data_config.img_size_h, data_config.img_size_w),
        train_ratio=data_config.train_ratio,
        test_ratio=data_config.test_ratio,
        seed=data_config.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    if args.model == "fcn":
        model = FCN(in_channels=3, out_channels=1, base_channels=args.base_channels).to(device)
    else:
        model = build_model(args.model, base_channels=args.base_channels).to(device)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=str(device))
    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    if 'best_val_defect_dice' in checkpoint:
        print(f"Best validation defect dice: {checkpoint['best_val_defect_dice']:.4f}")
    
    criterion = BCEDiceLoss()
    test_result = evaluate_model(model, test_loader, criterion, device, args.threshold)
    print(f"\nFinal Test Defect Dice: {test_result.metrics_defect_only.dice:.4f}")

if __name__ == "__main__":
    main()