"""Download Kaggle dataset into a target folder.

Usage (Colab):
  1) Upload kaggle.json
  2) pip install kaggle
  3) python scripts/download_kaggle_dataset.py --dataset girish17019/mobile-phone-defect-segmentation-dataset --out_dir /content --unzip
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
import zipfile

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--kaggle_json", type=str, default="/content/drive/MyDrive/likelion/kaggle.json")
    p.add_argument("--out_dir", type=str, default="/content")
    p.add_argument("--unzip", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kj = Path(args.kaggle_json)
    if not kj.exists():
        raise FileNotFoundError(f"kaggle.json not found at: {kj}")
    target = kaggle_dir / "kaggle.json"
    target.write_bytes(kj.read_bytes())
    os.chmod(target, 0o600)

    subprocess.check_call(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out_dir)])
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError("No zip downloaded.")
    zip_path = zips[0]
    print("Downloaded:", zip_path)

    if args.unzip:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        print("Extracted to:", out_dir)

if __name__ == "__main__":
    main()
