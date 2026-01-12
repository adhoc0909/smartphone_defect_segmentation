import torch
from pathlib import Path

# === 입력/출력 경로 ===
pth_path = Path("UNet_data-600_param2_best.pt")          # ← 네 pth 파일
pt_path  = Path("unet_aug2_600_param2.pt")     # ← 새로 만들 pt 파일

# === pth 로드 ===
ckpt = torch.load(pth_path, map_location="cpu")

if not isinstance(ckpt, dict):
    raise ValueError("pth 파일이 dict 형태가 아닙니다.")

if "model_state_dict" not in ckpt:
    raise ValueError("pth 파일에 'model_state_dict' 키가 없습니다.")

# === 기존 pt 포맷에 맞게 재구성 ===
new_ckpt = {
    "epoch": -1,                          # 정보 없으니 -1
    "model_state": ckpt["model_state_dict"],
    "optim_state": None,                  # 학습 끝났으므로 불필요
    "best_val_defect_dice": None,         # 없으면 None
}

# === 저장 ===
torch.save(new_ckpt, pt_path)

print(f"✅ 변환 완료: {pt_path.resolve()}")