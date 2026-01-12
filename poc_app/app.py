import sys
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import torch

# =========================================================
# Path setup (segtool import)
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJ_ROOT = APP_DIR.parent
SRC_DIR = PROJ_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from segtool.models_factory import build_model

def list_weight_files(weight_dir: Path):
    if not weight_dir.exists():
        return []
    return sorted([p.name for p in weight_dir.glob("*.pt")])

# =========================================================
# Checkpoint -> state_dict (your format: model_state)
# =========================================================
def load_state_dict(weight_path: str, device: str) -> dict:
    ckpt = torch.load(weight_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict.")
    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        return ckpt["model_state"]
    raise ValueError("No 'model_state' found in checkpoint.")


# =========================================================
# Build model with base_channel (try several arg names)
# =========================================================
def build_model_with_base(model_name: str, base_channel: int):
    # try common signatures
    for kwargs in [
        {"base_channel": base_channel},
        {"base_ch": base_channel},
        {"ch": base_channel},
        {"base_channels": base_channel},
        {"init_ch": base_channel},
    ]:
        # build_model(model_name, **kwargs)
        try:
            return build_model(model_name, **kwargs)
        except TypeError:
            pass
        # build_model(name=model_name, **kwargs)
        try:
            return build_model(name=model_name, **kwargs)
        except TypeError:
            pass
        # build_model(model_name=model_name, **kwargs)
        try:
            return build_model(model_name=model_name, **kwargs)
        except TypeError:
            pass

    # fallback: no base_channel arg
    try:
        return build_model(model_name)
    except TypeError:
        try:
            return build_model(name=model_name)
        except TypeError:
            return build_model(model_name=model_name)


@st.cache_resource
def load_model(model_name: str, base_channel: int, weight_path: str, device: str, strict_load: bool):
    model = build_model_with_base(model_name, base_channel)
    sd = load_state_dict(weight_path, device)
    missing, unexpected = model.load_state_dict(sd, strict=strict_load)
    model.to(device).eval()
    return model, missing, unexpected


# =========================================================
# EXACT inference logic (copy of your notebook logic)
# =========================================================
def infer_like_yours(model, orig_pil: Image.Image, img_w: int, img_h: int, threshold: float, alpha: float, device: str):
    model.eval()

    # 4) original
    orig_np = np.asarray(orig_pil).astype(np.uint8)
    H0, W0 = orig_np.shape[:2]

    # 5) preprocess (PIL resize + /255)
    inp_pil = orig_pil.resize((img_w, img_h), Image.BILINEAR)
    inp_np = (np.asarray(inp_pil).astype(np.float32) / 255.0)  # (H,W,3)
    x = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    # 6) inference (expects logits (1,1,H,W))
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        # enforce (1,1,H,W) expectation
        if not (logits.ndim == 4 and logits.shape[0] == 1 and logits.shape[1] == 1):
            # still try to adapt minimally to avoid silent failure
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)} (expected (1,1,H,W))")

        prob = torch.sigmoid(logits)[0, 0]        # (H,W)
        pred = (prob >= threshold).float()

    prob_np = prob.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    # 7) resize mask back to original
    mask_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
    mask_resized = mask_pil.resize((W0, H0), Image.NEAREST)
    mask_np = (np.asarray(mask_resized) > 0)

    # 8) overlay
    overlay = orig_np.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    overlay[mask_np] = ((1 - alpha) * overlay[mask_np] + alpha * red).astype(np.uint8)

    debug = {
        "orig_size": (H0, W0),
        "input_size": (img_h, img_w),
        "logits_shape": tuple(logits.shape),
        "prob_min": float(prob_np.min()),
        "prob_max": float(prob_np.max()),
        "prob_mean": float(prob_np.mean()),
        "pos_pixels_model_res": int((prob_np >= threshold).sum()),
        "pos_pixels_orig_res": int(mask_np.sum()),
    }

    return orig_np, prob_np, mask_np, overlay, debug


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Smartphone Defect Segmentation PoC", layout="wide")
st.title("📱 Smartphone Defect Segmentation - Streamlit PoC")

with st.sidebar:
    st.header("설정")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: **{device}**")

    model_name = st.text_input("model_name (models_factory 기준)", value="unet")
    base_channel = st.number_input("base_channel", value=64, min_value=4, step=4)

    
    weight_dir = APP_DIR / "weights"
    weight_files = list_weight_files(weight_dir)

    if len(weight_files) == 0:
        st.error("❌ weights 폴더에 .pt 파일이 없습니다.")
        st.stop()

    weight_file_name = st.selectbox(
        "weight 파일 선택",
        weight_files,
        help="poc_app/weights 폴더에 있는 pt 파일 목록"
    )

    weight_path = weight_dir / weight_file_name
    st.caption(f"📦 선택된 weight: {weight_file_name}")

    img_w = st.number_input("img_w (train width)", value=512, step=16)
    img_h = st.number_input("img_h (train height)", value=288, step=16)

    threshold = st.slider("threshold", 0.01, 0.95, 0.10, 0.01)
    alpha = st.slider("overlay alpha", 0.10, 0.90, 0.40, 0.05)

    strict_load = st.checkbox("strict state_dict load", value=True)

    if st.button("모델 캐시 삭제 & 재로드"):
        st.cache_resource.clear()
        st.rerun()

    show_debug = st.checkbox("디버그 정보 보기", value=True)

uploaded = st.file_uploader("📸 사진 업로드", type=["jpg", "jpeg", "png", "heic"])

if uploaded is None:
    st.info("사진을 업로드하면 segmentation 결과가 표시됩니다.")
    st.stop()

try:
    orig_pil = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"이미지 로드 실패: {e}\n\nHEIC면 JPG로 변환해서 업로드해보세요.")
    st.stop()

# Load model
try:
    model, missing, unexpected = load_model(model_name, int(base_channel), weight_path, device, strict_load)
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

if show_debug:
    with st.expander("로드 정보", expanded=False):
        st.write("missing keys count:", len(missing))
        st.write("unexpected keys count:", len(unexpected))
        if len(missing) > 0:
            st.write("missing sample:", missing[:30])
        if len(unexpected) > 0:
            st.write("unexpected sample:", unexpected[:30])

# Inference (exactly your logic)
try:
    orig_np, prob_np, mask_np, overlay, dbg = infer_like_yours(
        model=model,
        orig_pil=orig_pil,
        img_w=int(img_w),
        img_h=int(img_h),
        threshold=float(threshold),
        alpha=float(alpha),
        device=device,
    )
except Exception as e:
    st.error(f"추론 실패: {e}")
    st.stop()

if show_debug:
    with st.expander("Inference Debug", expanded=True):
        for k, v in dbg.items():
            st.write(f"{k}: {v}")

# Display
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Original")
    st.image(orig_np, use_container_width=True)
with c2:
    st.subheader("Prob (model resolution)")
    st.image((prob_np * 255).astype(np.uint8), clamp=True, use_container_width=True)
with c3:
    st.subheader("Overlay (original resolution)")
    st.image(overlay, use_container_width=True)

st.caption("팁: threshold를 낮추면 더 많이 잡히지만 FP가 늘 수 있어요.")
