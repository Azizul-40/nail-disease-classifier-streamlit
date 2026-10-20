"""
নখের রোগ শনাক্তকরণ -- Streamlit demo app
STEP37-এ lock করা দুইটা final model ব্যবহার করে:
  - Dataset-1 (৬ ক্লাস): STEP26 -- ConvNeXt-Tiny + CBAM
  - Dataset-2 (৩ ক্লাস): STEP23B -- ConvNeXt-Tiny standalone

চালানোর আগে:
    pip install -r requirements.txt
    streamlit run app.py
প্রথমবার চালালে Google Drive থেকে দুইটা checkpoint (~110MB করে) auto-download হবে।
"""

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model_defs import (
    DRIVE_FILE_IDS,
    IMAGE_SIZE,
    LABEL_MAPS,
    LOCKED_SHA256,
    MODEL_BUILDERS,
    MODEL_NAMES,
)

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_of(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint(ds_name: str) -> Path:
    """Download the locked checkpoint from Drive if it's not already present locally."""
    ckpt_path = CKPT_DIR / f"{ds_name}_final_locked.pt"
    if not ckpt_path.exists():
        import gdown

        file_id = DRIVE_FILE_IDS[ds_name]
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"{ds_name}-এর locked checkpoint ডাউনলোড হচ্ছে (~110MB, প্রথমবার একটু সময় লাগবে)..."):
            gdown.download(url, str(ckpt_path), quiet=False)
    return ckpt_path


@st.cache_resource(show_spinner=False)
def load_model(ds_name: str):
    ckpt_path = ensure_checkpoint(ds_name)

    # Optional but recommended: verify the checkpoint matches STEP37's lock record
    # before loading, so the app never silently serves a corrupted/wrong file.
    actual_sha = sha256_of(ckpt_path)
    expected_sha = LOCKED_SHA256[ds_name]
    integrity_ok = actual_sha.lower() == expected_sha.lower()

    label_map = LABEL_MAPS[ds_name]
    class_names = sorted(label_map.keys(), key=lambda c: label_map[c])
    num_classes = len(class_names)

    model = MODEL_BUILDERS[ds_name](num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model", ckpt))) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model, class_names, integrity_ok


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Exactly mirrors the pipeline's own inference preprocessing (STEP41):
    resize to 224x224, RGB uint8 -> float32 [0,1], CHW, batch dim. Model
    normalizes internally, so no further normalization here."""
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict(model, class_names, pil_image: Image.Image):
    x = preprocess(pil_image).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), dict(zip(class_names, probs.tolist()))


# ============================== UI ==============================

st.set_page_config(page_title="নখের রোগ শনাক্তকরণ", page_icon="🩺", layout="centered")

st.title("🩺 নখের রোগ শনাক্তকরণ")
st.caption("STEP37-locked final models | Research demo -- clinical diagnosis-এর বিকল্প নয়")

ds_choice = st.selectbox(
    "কোন dataset / model ব্যবহার করবে?",
    options=["dataset_1", "dataset_2"],
    format_func=lambda d: MODEL_NAMES[d],
)

uploaded = st.file_uploader("নখের ছবি আপলোড করো (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="আপলোড করা ছবি", use_container_width=True)

    with st.spinner("মডেল লোড হচ্ছে..."):
        model, class_names, integrity_ok = load_model(ds_choice)

    if not integrity_ok:
        st.warning(
            "⚠️ এই checkpoint-এর SHA256, STEP37-এর lock record-এর সাথে মিলছে না -- "
            "ডাউনলোড ঠিকমতো হয়নি হতে পারে। checkpoints/ ফোল্ডার থেকে ফাইলটা ডিলিট করে আবার চালাও।"
        )

    with st.spinner("Predict করা হচ্ছে..."):
        pred_class, pred_conf, all_probs = predict(model, class_names, image)

    st.subheader(f"🔎 Prediction: **{pred_class}**")
    st.metric("Confidence", f"{pred_conf * 100:.2f}%")

    st.write("সব ক্লাসের probability:")
    sorted_probs = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))
    st.bar_chart(sorted_probs)

    if pred_conf < 0.60:
        st.info("কনফিডেন্স তুলনামূলক কম -- এই ছবিটা ambiguous/uncertain case হতে পারে।")
else:
    st.info("একটা নখের ছবি আপলোড করলে prediction দেখাবো।")

st.divider()
st.caption(
    "⚠️ এই টুলটা একটা research demo, STEP52-এ দেখা গেছে cross-dataset generalization "
    "(বিশেষ করে Healthy ক্লাসে) এখনো দুর্বল, আর STEP59-এ দেখা গেছে noisy ছবিতে accuracy কমে যায়। "
    "চূড়ান্ত ক্লিনিক্যাল সিদ্ধান্তের জন্য ব্যবহার করা ঠিক না।"
)
