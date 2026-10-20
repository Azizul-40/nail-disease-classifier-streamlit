"""
NailDx AI -- নখের রোগ শনাক্তকরণ (Streamlit app)

STEP37-locked models:
  - Dataset-1 (6 classes): STEP26 -- ConvNeXt-Tiny + CBAM
  - Dataset-2 (3 classes): STEP23B -- ConvNeXt-Tiny standalone

Features: bilingual UI (bn/en), light/dark theme, patient-info gate before
prediction, multi-image upload + camera capture, optional dual-model
comparison, Grad-CAM heatmap overlay, PDF prescription-style report
download, and a local patient archive.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import hashlib
import io
import json
import uuid
from datetime import datetime
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
)

APP_DIR = Path(__file__).parent
CKPT_DIR = APP_DIR / "checkpoints"
DATA_DIR = APP_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "archive.json"
CKPT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACCENT_START, ACCENT_END = "#0D7D73", "#0E9BC9"  # teal -> sky blue (app theme)

DISEASE_TYPES_D1 = [
    "Acral_Lentiginous_Melanoma", "Healthy_Nail", "Onychogryphosis",
    "blue_finger", "clubbing", "pitting",
]
DISEASE_TYPES_D2 = ["healthy", "onychomycosis", "psoriasis"]

# Clean, user-facing model names -- internal checkpoint/step codenames
# (STEP26, STEP23B, etc. -- see model_defs.MODEL_NAMES) are never shown in the UI.
MODEL_UI_LABEL = {
    "dataset_1": {"bn": "মডেল ১ — CBAM Attention (৬ ধরনের রোগ)", "en": "Model 1 — CBAM Attention (6 conditions)"},
    "dataset_2": {"bn": "মডেল ২ — স্ট্যান্ডার্ড (৩ ধরনের রোগ)", "en": "Model 2 — Standard (3 conditions)"},
}


def model_label(ds_name: str) -> str:
    return MODEL_UI_LABEL[ds_name][st.session_state.lang]

# Short, general, non-diagnostic guidance per predicted class. General
# wording only (see "About" tab -> Sources for where this kind of
# guidance is drawn from: WHO public skin-health information and
# DermNet NZ, a widely used clinician/patient dermatology reference).
ADVISORY = {
    "bn": {
        "Healthy_Nail": "নখ স্বাভাবিক দেখাচ্ছে। তবুও নখে কোনো পরিবর্তন খেয়াল করলে ডাক্তার দেখানো ভালো।",
        "healthy": "নখ স্বাভাবিক দেখাচ্ছে। তবুও নখে কোনো পরিবর্তন খেয়াল করলে ডাক্তার দেখানো ভালো।",
        "_default": "এই prediction নিশ্চিত হওয়ার জন্য একজন ডার্মাটোলজিস্ট-এর সাথে পরামর্শ করা উচিত। এটি শুধু সাধারণ নির্দেশনা, চূড়ান্ত চিকিৎসা পরামর্শ না।",
    },
    "en": {
        "Healthy_Nail": "The nail appears healthy. Still, see a doctor if you notice any new changes.",
        "healthy": "The nail appears healthy. Still, see a doctor if you notice any new changes.",
        "_default": "Consulting a dermatologist is recommended to confirm this prediction. This is general guidance only, not a final medical decision.",
    },
}

# --------------------------------------------------------------------------
# i18n
# --------------------------------------------------------------------------
T = {
    "bn": {
        "page_title": "নখের রোগ শনাক্তকরণ", "page_sub": "ConvNeXt-Tiny ভিত্তিক গবেষণা প্রোটোটাইপ — ডায়াগনস্টিক টুল না।",
        "settings": "সেটিংস", "lang_btn": "English", "theme_btn_dark": "🌙 Dark", "theme_btn_light": "☀️ Light",
        "our_models": "আমাদের মডেল", "disease_types": "শনাক্তযোগ্য রোগ",
        "disclaimer_title": "গবেষণা প্রকল্প",
        "disclaimer_body": "এটি একটি research demo, ক্লিনিক্যাল সিদ্ধান্তের জন্য ব্যবহার করা ঠিক না।",
        "patient_title": "রোগীর তথ্য", "patient_hint": "প্রথমে রোগীর তথ্য পূরণ করো, এরপর ছবি আপলোড করা যাবে।",
        "f_name": "নাম *", "f_nid": "NID নম্বর *", "f_mobile": "মোবাইল নম্বর *",
        "f_passport": "পাসপোর্ট নম্বর (ঐচ্ছিক)", "f_age": "বয়স *", "f_sex": "লিঙ্গ *", "f_doctor": "ডাক্তারের নাম *",
        "sex_m": "পুরুষ", "sex_f": "মহিলা", "sex_o": "অন্যান্য",
        "submit_info": "তথ্য জমা দিন", "edit_info": "✎ তথ্য বদলান",
        "missing_fields": "⚠️ অনুগ্রহ করে * চিহ্নিত সব তথ্য পূরণ করো।",
        "info_saved": "✓ তথ্য সংরক্ষিত হয়েছে", "locked_hint": "🔒 অনুগ্রহ করে প্রথমে 'রোগীর তথ্য' ট্যাবে তথ্য পূরণ করো",
        "upload_locked_warn": "⚠️ ছবি আপলোডের আগে 'রোগীর তথ্য' ট্যাব থেকে রোগীর তথ্য পূরণ করে জমা দাও।",
        "select_model": "কোন dataset / model ব্যবহার করবে?",
        "compare_toggle": "⚖ দুই মডেল দিয়ে Compare করো",
        "upload_label": "নখের ছবি আপলোড করো (একাধিক ছবি দেওয়া যাবে)",
        "camera_label": "📷 অথবা ক্যামেরা দিয়ে ছবি তোলো",
        "add_camera_btn": "ছবিটা যোগ করো",
        "predict_btn": "সব ছবি Predict করো", "predicting": "Predict করা হচ্ছে...",
        "loading_model": "মডেল লোড হচ্ছে...",
        "results_heading": "প্রতিটা ছবির ফলাফল", "prediction": "Prediction", "confidence": "Confidence",
        "low_conf": "কম নিশ্চয়তা", "remove": "✕ বাদ দাও", "heatmap_toggle": "🔥 মডেল কোথায় তাকিয়ে prediction দিয়েছে (heatmap)",
        "advice_title": "💡 AI পরামর্শ (verified source ভিত্তিক)",
        "no_images_yet": "একটা বা একাধিক নখের ছবি আপলোড করে Predict চাপো — প্রতিটার ফলাফল এখানে নম্বর দিয়ে দেখাবো।",
        "download_title": "📄 রিপোর্ট প্রিভিউ", "download_btn": "⬇ রিপোর্ট ডাউনলোড করো (PDF)",
        "no_report_yet": "এখনো কোনো prediction করা হয়নি। প্রথমে Prediction ট্যাব থেকে ছবি Predict করো।",
        "archive_search": "রোগীর নাম দিয়ে খুঁজো...", "archive_empty": "আর্কাইভ এখনো খালি।",
        "archive_view": "রিপোর্ট দেখো / ডাউনলোড করো", "archive_remove": "✕ মুছে ফেলো",
        "about_heading": "মডেল কীভাবে কাজ করে",
        "about_body": "দুটি লক করা ConvNeXt-Tiny মডেল ব্যবহার করা হয়েছে — একটি CBAM attention সহ (৬-ক্লাস, accuracy 96.61%), অন্যটি standalone (৩-ক্লাস, accuracy 94.88%)। প্রতিটি checkpoint লোড করার আগে SHA256 দিয়ে integrity verify করা হয়, যাতে ভুল বা পরিবর্তিত ফাইল চলে না আসে।",
        "dataset_heading": "ডেটাসেট",
        "dataset_body": "নখের ছবি থেকে বিভিন্ন কন্ডিশন (যেমন healthy nail, psoriasis, onychomycosis, clubbing, pitting ইত্যাদি) শনাক্ত করার জন্য ট্রেইন করা হয়েছে। প্রোডাকশনে ব্যবহৃত মডেল দুটি ট্রেইনিং শেষে ফাইনাল হিসেবে লক করা — ভবিষ্যতে নতুন ডেটা দিয়ে আবার রি-ট্রেইন করা যাবে।",
        "source_heading": "তথ্যসূত্র",
        "source_body": "রিপোর্টে যুক্ত সাধারণ স্বাস্থ্য-পরামর্শ WHO-এর সাধারণ ত্বক-স্বাস্থ্য তথ্য এবং DermNet NZ (একটি স্বীকৃত ডার্মাটোলজি রেফারেন্স) থেকে অনুপ্রাণিত। এটি কোনো ব্যক্তিগত চিকিৎসা পরামর্শ না।",
        "source_link_who": "WHO ফ্যাক্ট শিট দেখো", "source_link_dermnet": "DermNet NZ দেখো",
    },
    "en": {
        "page_title": "Nail Disease Classifier", "page_sub": "ConvNeXt-Tiny based research prototype — not a diagnostic tool.",
        "settings": "Settings", "lang_btn": "বাংলা", "theme_btn_dark": "🌙 Dark", "theme_btn_light": "☀️ Light",
        "our_models": "Our Models", "disease_types": "Detectable Diseases",
        "disclaimer_title": "Research project",
        "disclaimer_body": "This is a research demo — not for clinical decision-making.",
        "patient_title": "Patient Info", "patient_hint": "Fill patient details first, then you can upload a photo.",
        "f_name": "Name *", "f_nid": "NID number *", "f_mobile": "Mobile number *",
        "f_passport": "Passport number (optional)", "f_age": "Age *", "f_sex": "Sex *", "f_doctor": "Doctor's name *",
        "sex_m": "Male", "sex_f": "Female", "sex_o": "Other",
        "submit_info": "Submit info", "edit_info": "✎ Edit info",
        "missing_fields": "⚠️ Please fill in all fields marked with *.",
        "info_saved": "✓ Info saved", "locked_hint": "🔒 Please fill patient info in the 'Patient Info' tab first",
        "upload_locked_warn": "⚠️ Fill in patient info in the 'Patient Info' tab before uploading an image.",
        "select_model": "Which dataset / model do you want to use?",
        "compare_toggle": "⚖ Compare both models",
        "upload_label": "Upload nail photo(s) (multiple allowed)",
        "camera_label": "📷 Or take a photo with your camera",
        "add_camera_btn": "Add this photo",
        "predict_btn": "Predict all images", "predicting": "Predicting...",
        "loading_model": "Loading model...",
        "results_heading": "Result per image", "prediction": "Prediction", "confidence": "Confidence",
        "low_conf": "Low confidence", "remove": "✕ Remove", "heatmap_toggle": "🔥 See where the model looked (heatmap)",
        "advice_title": "💡 AI advisory (from a verified source)",
        "no_images_yet": "Upload one or more nail photos and press Predict — each image's result appears here, numbered.",
        "download_title": "📄 Report preview", "download_btn": "⬇ Download report (PDF)",
        "no_report_yet": "No prediction yet. Go to the Prediction tab and predict an image first.",
        "archive_search": "Search by patient name...", "archive_empty": "Archive is empty.",
        "archive_view": "View / download report", "archive_remove": "✕ Remove",
        "about_heading": "How the models work",
        "about_body": "Two locked ConvNeXt-Tiny models are used — one with CBAM attention (6-class, 96.61% accuracy), the other standalone (3-class, 94.88% accuracy). Every checkpoint is SHA256-verified before loading, so a wrong or modified file is never silently used.",
        "dataset_heading": "Dataset",
        "dataset_body": "Trained to recognize several nail conditions (healthy nail, psoriasis, onychomycosis, clubbing, pitting, and more). The models shipped in this app are the final, locked versions from training -- they can be retrained on new data in the future.",
        "source_heading": "Sources",
        "source_body": "The general health guidance added to reports is drawn from WHO's public skin-health information and DermNet NZ (a widely used dermatology reference). It is not personal medical advice.",
        "source_link_who": "View WHO fact sheets", "source_link_dermnet": "View DermNet NZ",
    },
}


def tr(key: str) -> str:
    return T[st.session_state.lang][key]


def pretty_label(label: str) -> str:
    """Display-only formatting: 'Healthy_Nail' -> 'Healthy Nail'. Never used for
    model/archive lookups -- those keep the raw class name."""
    return label.replace("_", " ").strip()


# --------------------------------------------------------------------------
# Model loading / inference
# --------------------------------------------------------------------------

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
    ckpt_path = CKPT_DIR / f"{ds_name}_final_locked.pt"
    if not ckpt_path.exists():
        import gdown

        file_id = DRIVE_FILE_IDS[ds_name]
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"{ds_name} checkpoint downloading (~110MB, first run only)..."):
            try:
                gdown.download(url, str(ckpt_path), quiet=False, fuzzy=True)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"❌ Couldn't download the {ds_name} checkpoint from Google Drive "
                    f"({exc}). Make sure the file is shared as \"Anyone with the link — Viewer\"."
                )
                raise
    return ckpt_path


@st.cache_resource(show_spinner=False)
def load_model(ds_name: str):
    ckpt_path = ensure_checkpoint(ds_name)
    actual_sha = sha256_of(ckpt_path)
    expected_sha = LOCKED_SHA256[ds_name]
    integrity_ok = actual_sha.lower() == expected_sha.lower()

    label_map = LABEL_MAPS[ds_name]
    class_names = sorted(label_map.keys(), key=lambda c: label_map[c])
    num_classes = len(class_names)

    model = MODEL_BUILDERS[ds_name](num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = (
        ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
        if isinstance(ckpt, dict)
        else ckpt
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, class_names, integrity_ok


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def predict(model, class_names, pil_image: Image.Image):
    with torch.no_grad():
        x = preprocess(pil_image).to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), dict(zip(class_names, probs.tolist()))


def gradcam_feature_module(model, ds_name: str):
    """The last conv-feature stage to hook for Grad-CAM, per model type."""
    if ds_name == "dataset_1":
        return model.features  # ConvNeXtCBAM
    return model.m.features  # ConvNeXtTFull


def compute_gradcam(model, ds_name: str, pil_image: Image.Image, class_idx: int) -> np.ndarray:
    """Returns a (IMAGE_SIZE, IMAGE_SIZE) array in [0, 1]."""
    feature_module = gradcam_feature_module(model, ds_name)
    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, out):
        activations["v"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["v"] = grad_out[0]

    h1 = feature_module.register_forward_hook(fwd_hook)
    h2 = feature_module.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        x = preprocess(pil_image).to(DEVICE).requires_grad_(False)
        logits = model(x)
        score = logits[0, class_idx]
        score.backward()
        acts = activations["v"][0].detach()
        grads = gradients["v"][0].detach()
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    return np.array(cam_img, dtype=np.float32) / 255.0


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha_max: float = 0.6) -> Image.Image:
    """Yellow -> red 'hot' overlay (classic Grad-CAM look), alpha scaled by activation."""
    base = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = 255.0  # R
    heat[..., 1] = 255.0 * (1.0 - cam)  # G: high where cam low (yellow), low where cam high (red)
    heat[..., 2] = 0.0
    alpha = np.clip(cam * alpha_max, 0.0, alpha_max)[..., None]
    blended = base * (1 - alpha) + heat * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def advisory_for(label: str) -> str:
    bucket = ADVISORY[st.session_state.lang]
    return bucket.get(label, bucket["_default"])


# --------------------------------------------------------------------------
# Archive (local JSON persistence)
# --------------------------------------------------------------------------

def load_archive():
    if ARCHIVE_PATH.exists():
        try:
            return json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
    return []


def save_archive(records):
    ARCHIVE_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_patient(patient: dict, results: list):
    records = st.session_state.archive
    summary = ", ".join(f"{pretty_label(r['label'])} ({r['pct']:.0f}%)" for r in results)
    records.insert(
        0,
        {
            "id": str(uuid.uuid4())[:8],
            "name": patient["name"],
            "meta": f"{patient['age']}y · {patient['sex']}",
            "doctor": patient["doctor"],
            "date": datetime.now().strftime("%d %b %Y, %H:%M"),
            "result_summary": summary,
            "patient": patient,
            "results": [{"label": r["label"], "pct": r["pct"], "model": r["model"]} for r in results],
        },
    )
    st.session_state.archive = records
    save_archive(records)


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------

def build_pdf_report(patient: dict, results: list, images: list) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NailDx AI -- Prediction Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, value in [
        ("Name", patient.get("name", "")),
        ("NID number", patient.get("nid", "")),
        ("Mobile number", patient.get("mobile", "")),
        ("Passport number", patient.get("passport", "") or "-"),
        ("Age / Sex", f"{patient.get('age', '')} / {patient.get('sex', '')}"),
        ("Doctor", patient.get("doctor", "")),
    ]:
        pdf.cell(0, 6, f"{label}: {value}", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Predictions", ln=True)
    for idx, (res, img) in enumerate(zip(results, images), start=1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"Image {idx}: {pretty_label(res['label'])}  ({res['pct']:.1f}%)  [{res['model']}]", ln=True)
        if res.get("second"):
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(
                0, 6,
                f"   Compare ({res['second']['model']}): {pretty_label(res['second']['label'])} ({res['second']['pct']:.1f}%)",
                ln=True,
            )
        try:
            buf = io.BytesIO()
            img.convert("RGB").resize((160, 120)).save(buf, format="JPEG")
            buf.seek(0)
            pdf.image(buf, w=50)
        except Exception:  # noqa: BLE001
            pass
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, advisory_for(res["label"]))
        pdf.ln(2)

    pdf.set_font("Helvetica", "I", 8)
    pdf.ln(4)
    pdf.multi_cell(
        0, 5,
        "This is a research prototype (STEP37-locked models). Predictions are not a "
        "clinical diagnosis. General guidance above is drawn from WHO public skin-health "
        "information and DermNet NZ -- see the 'About the models' tab for sources.",
    )
    return bytes(pdf.output(dest="S"))


# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------

defaults = {
    "lang": "en",
    "theme": "light",
    "patient": None,
    "patient_submitted": False,
    "compare": False,
    "images": [],  # list of {id, image}
    "results": [],  # list of {id, image, label, pct, model, low, second, cam}
    "sidebar_images_counter": 0,
    "archive": load_archive(),
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------------------------------
# Page config + theming
# --------------------------------------------------------------------------

st.set_page_config(page_title="NailDx AI", page_icon="🩺", layout="wide")

DARK = st.session_state.theme == "dark"
C_BG = "#101018" if DARK else "#F7F7FB"
C_PANEL = "#191922" if DARK else "#FFFFFF"
C_SUBTLE = "#20202B" if DARK else "#F6F5FB"
C_BORDER = "#2B2B38" if DARK else "#E8E7F2"
C_TEXT = "#F1F1F6" if DARK else "#1F2430"
C_MUTED = "#9A9AAC" if DARK else "#6B7280"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {C_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {C_PANEL}; }}
    section[data-testid="stSidebar"] * {{ color: {C_TEXT} !important; }}
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] * {{
        color: {C_TEXT} !important;
    }}
    div[data-testid="stMetricValue"] {{ color: {C_TEXT} !important; }}

    .stButton > button {{
        background: {C_SUBTLE}; color: {C_TEXT} !important;
        border: 1px solid {C_BORDER}; font-weight: 600;
    }}
    .stButton > button:hover {{ border-color: {ACCENT_START}; color: {ACCENT_START} !important; }}
    .stButton > button p {{ color: inherit !important; }}

    div[data-testid="stMarkdownContainer"] a, .ndx-card a {{ color: {ACCENT_START} !important; font-weight: 700; }}

    /* Bold, underlined-when-active tabs, matching the approved design */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {C_BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 700; font-size: 14.5px; color: {C_MUTED} !important;
        padding: 10px 4px; background: transparent;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: inherit !important; font-weight: 700; }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT_START} !important;
        border-bottom: 3px solid {ACCENT_START} !important;
    }}
    .stTabs [aria-selected="true"] p {{ color: {ACCENT_START} !important; text-decoration: underline; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT_START} !important; }}

    /* Card-style borders for st.container(border=True) blocks (result cards, archive rows) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {C_PANEL} !important; border: 1px solid {C_BORDER} !important;
        border-radius: 14px !important; padding: 4px;
    }}

    /* Rounded, card-style form inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 10px !important; background: {C_SUBTLE} !important;
        border: 1px solid {C_BORDER} !important; color: {C_TEXT} !important;
    }}
    .stForm {{
        background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 16px; padding: 18px 20px;
    }}

    .ndx-header {{
        background: linear-gradient(135deg, {ACCENT_START}, {ACCENT_END});
        padding: 22px 26px; border-radius: 16px; color: white; margin-bottom: 10px;
    }}
    .ndx-header h1 {{ margin: 0; font-size: 23px; font-weight: 800; color: white !important; }}
    .ndx-header p {{ margin: 4px 0 0; opacity: .92; font-size: 13.5px; color: white !important; }}

    .ndx-card {{
        background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 14px;
        padding: 15px 16px; margin-bottom: 14px;
    }}
    .ndx-card-title {{ font-weight: 800; font-size: 15px; color: {C_TEXT}; margin-bottom: 8px;
        display: flex; align-items: center; gap: 7px; }}
    .ndx-card hr {{ border: none; border-top: 1px solid {C_BORDER}; margin: 6px 0 10px; }}
    .ndx-model-row {{ display: flex; align-items: center; gap: 8px; font-weight: 700;
        font-size: 13.5px; color: {C_TEXT}; margin-bottom: 2px; }}
    .ndx-dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
    .ndx-model-sub {{ font-size: 12px; color: {C_MUTED}; margin: 0 0 10px 16px; line-height: 1.6; }}
    .ndx-disease-item {{ font-size: 13px; color: {C_MUTED}; margin: 4px 0 4px 2px; }}
    .ndx-disease-dot {{ color: {ACCENT_START}; margin-right: 7px; }}
    .ndx-disclaimer {{
        background: {C_SUBTLE}; border: 1px solid {C_BORDER}; border-left: 3px solid {ACCENT_START};
        border-radius: 12px; padding: 13px 14px;
    }}
    .ndx-disclaimer-title {{ font-weight: 800; font-size: 13px; color: {C_TEXT}; margin-bottom: 4px; }}
    .ndx-disclaimer-body {{ font-size: 12px; line-height: 1.6; color: {C_MUTED}; }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {ACCENT_START}, {ACCENT_END}); border: none;
    }}

    /* Kill the browser's harsh focus-ring rectangle on tabs/buttons -- design has no box outline */
    .stTabs button:focus, .stTabs button:focus-visible,
    .stButton > button:focus, .stButton > button:focus-visible {{
        outline: none !important; box-shadow: none !important;
    }}

    /* Section header pill (e.g. "Patient Info" title), matching the approved design */
    .ndx-pill {{
        display: inline-block; background: linear-gradient(135deg, {ACCENT_START}, {ACCENT_END});
        color: white !important; font-weight: 800; font-size: 14px;
        padding: 9px 22px; border-radius: 999px; margin-bottom: 16px;
    }}
    .ndx-pill * {{ color: white !important; }}

    /* Colored panel behind any uploaded/result thumbnail image */
    div[data-testid="column"]:has(img) {{
        background: {C_SUBTLE}; border-radius: 12px; padding: 10px;
    }}
    .ndx-badge-num {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%;
        background: linear-gradient(135deg, {ACCENT_START}, {ACCENT_END});
        color: white !important; font-weight: 800; font-size: 12.5px; margin-bottom: 6px;
    }}

    /* Archive row: circular initial avatar + colored result pill */
    .ndx-avatar {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 34px; height: 34px; border-radius: 50%; background: {C_SUBTLE};
        border: 1px solid {C_BORDER}; color: {ACCENT_START} !important; font-weight: 800;
        font-size: 14px; margin-right: 10px; flex: none;
    }}
    .ndx-archive-row {{ display: flex; align-items: center; }}
    .ndx-archive-name {{ font-weight: 700; color: {C_TEXT}; }}
    .ndx-archive-meta {{ font-size: 12px; color: {C_MUTED}; }}
    .ndx-result-pill {{
        font-weight: 700; font-size: 12.5px; color: {ACCENT_START} !important;
        background: {C_SUBTLE}; border: 1px solid {C_BORDER}; border-radius: 999px;
        padding: 4px 12px; display: inline-block;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="ndx-header"><h1>🩺 {tr('page_title')}</h1><p>{tr('page_sub')}</p></div>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 NailDx AI")
    c1, c2 = st.columns(2)
    if c1.button(tr("lang_btn"), use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "bn" else "bn"
        st.rerun()
    theme_label = tr("theme_btn_dark") if st.session_state.theme == "light" else tr("theme_btn_light")
    if c2.button(theme_label, use_container_width=True):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

    st.markdown(
        f"""
        <div class="ndx-card">
          <div class="ndx-card-title">🧬 {tr('our_models')}</div>
          <hr>
          <div class="ndx-model-row"><span class="ndx-dot" style="background:{ACCENT_START}"></span>Dataset-1</div>
          <div class="ndx-model-sub">ConvNeXt-Tiny + CBAM — 6 classes<br>Accuracy 96.61%</div>
          <div class="ndx-model-row"><span class="ndx-dot" style="background:{ACCENT_END}"></span>Dataset-2</div>
          <div class="ndx-model-sub">ConvNeXt-Tiny standalone — 3 classes<br>Accuracy 94.88%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    disease_html = "".join(f'<div class="ndx-disease-item"><span class="ndx-disease-dot">●</span>{d}</div>' for d in DISEASE_TYPES_D1)
    st.markdown(
        f"""
        <div class="ndx-card">
          <div class="ndx-card-title">🔬 {tr('disease_types')} (6 types)</div>
          <hr>
          {disease_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="ndx-disclaimer">
          <div class="ndx-disclaimer-title">⚠ {tr('disclaimer_title')}</div>
          <div class="ndx-disclaimer-body">{tr('disclaimer_body')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

# Tab labels stay fixed in English regardless of the language toggle.
tab_predict, tab_patient, tab_download, tab_archive, tab_about = st.tabs(
    ["Prediction", "Patient Info", "Download", "Archive", "About the models"]
)

# ---------------- Patient Info tab ----------------
with tab_patient:
    if not st.session_state.patient_submitted:
        st.markdown(f'<div class="ndx-pill">👤 {tr("patient_title")}</div>', unsafe_allow_html=True)
        st.caption(tr("patient_hint"))
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            name = col1.text_input(tr("f_name"), value=(st.session_state.patient or {}).get("name", ""))
            nid = col2.text_input(tr("f_nid"), value=(st.session_state.patient or {}).get("nid", ""))
            mobile = col1.text_input(tr("f_mobile"), value=(st.session_state.patient or {}).get("mobile", ""))
            passport = col2.text_input(tr("f_passport"), value=(st.session_state.patient or {}).get("passport", ""))
            age = col1.text_input(tr("f_age"), value=(st.session_state.patient or {}).get("age", ""))
            sex_options = [tr("sex_m"), tr("sex_f"), tr("sex_o")]
            sex = col2.selectbox(tr("f_sex"), options=sex_options)
            doctor = st.text_input(tr("f_doctor"), value=(st.session_state.patient or {}).get("doctor", ""))

            submitted = st.form_submit_button(tr("submit_info"), type="primary", use_container_width=True)
            if submitted:
                required_ok = all([name.strip(), nid.strip(), mobile.strip(), age.strip(), doctor.strip()])
                if not required_ok:
                    st.error(tr("missing_fields"))
                else:
                    st.session_state.patient = {
                        "name": name.strip(), "nid": nid.strip(), "mobile": mobile.strip(),
                        "passport": passport.strip(), "age": age.strip(), "sex": sex, "doctor": doctor.strip(),
                    }
                    st.session_state.patient_submitted = True
                    st.rerun()
    else:
        p = st.session_state.patient
        st.success(tr("info_saved"))
        st.write(f"**{tr('f_name')[:-2]}:** {p['name']}  |  **{tr('f_age')[:-2]}/{tr('f_sex')[:-2]}:** {p['age']}/{p['sex']}")
        st.write(f"**{tr('f_doctor')[:-2]}:** {p['doctor']}")
        if st.button(tr("edit_info")):
            st.session_state.patient_submitted = False
            st.rerun()

# ---------------- Prediction tab ----------------
with tab_predict:
    if not st.session_state.patient_submitted:
        st.info(tr("locked_hint"))

    ds_choice = st.selectbox(tr("select_model"), options=["dataset_1", "dataset_2"], format_func=model_label)
    st.session_state.compare = st.toggle(tr("compare_toggle"), value=st.session_state.compare)

    uploaded_files = st.file_uploader(
        tr("upload_label"), type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        disabled=not st.session_state.patient_submitted,
    )
    cam_photo = st.camera_input(tr("camera_label"), disabled=not st.session_state.patient_submitted)

    if (uploaded_files or cam_photo is not None) and not st.session_state.patient_submitted:
        st.warning(tr("upload_locked_warn"))
        uploaded_files, cam_photo = None, None

    pending_images = []
    if uploaded_files:
        for f in uploaded_files:
            pending_images.append(Image.open(f))
    if cam_photo is not None:
        pending_images.append(Image.open(cam_photo))

    predict_clicked = st.button(tr("predict_btn"), type="primary", use_container_width=True,
                                 disabled=not st.session_state.patient_submitted or not pending_images)

    if predict_clicked and pending_images:
        with st.spinner(tr("loading_model")):
            model, class_names, integrity_ok = load_model(ds_choice)
        if not integrity_ok:
            st.warning("⚠️ Checkpoint SHA256 does not match the STEP37 lock record.")

        second_model, second_class_names = None, None
        if st.session_state.compare:
            other_ds = "dataset_2" if ds_choice == "dataset_1" else "dataset_1"
            with st.spinner(tr("loading_model")):
                second_model, second_class_names, _ = load_model(other_ds)

        results = []
        with st.spinner(tr("predicting")):
            for img in pending_images:
                label, conf, _ = predict(model, class_names, img)
                entry = {
                    "id": str(uuid.uuid4())[:8], "image": img, "label": label, "pct": conf * 100,
                    "model": model_label(ds_choice), "low": conf < 0.5, "cam": None,
                }
                if second_model is not None:
                    other_ds = "dataset_2" if ds_choice == "dataset_1" else "dataset_1"
                    label2, conf2, _ = predict(second_model, second_class_names, img)
                    entry["second"] = {
                        "label": label2, "pct": conf2 * 100, "model": model_label(other_ds),
                    }
                results.append(entry)
        st.session_state.results = results
        st.session_state.results_ds_choice = ds_choice
        archive_patient(st.session_state.patient, results)

    if not st.session_state.results:
        st.info(tr("no_images_yet"))
    else:
        st.markdown(f"**{tr('results_heading')}**")
        keep = []
        for i, r in enumerate(st.session_state.results, start=1):
            with st.container(border=True):
                col_img, col_body = st.columns([1, 2])
                col_img.markdown(f'<div class="ndx-badge-num">{i}</div>', unsafe_allow_html=True)
                col_img.image(r["image"], use_container_width=True)
                with col_body:
                    badge = f" :red[{tr('low_conf')}]" if r["low"] else ""
                    st.markdown(f"🔎 {tr('prediction')} ({r['model']}){badge}")
                    st.markdown(f"### {pretty_label(r['label'])}")
                    st.progress(min(r["pct"] / 100, 1.0), text=f"{r['pct']:.1f}%")
                    if r.get("second"):
                        st.markdown(f"🔎 {tr('prediction')} ({r['second']['model']})")
                        st.markdown(f"**{pretty_label(r['second']['label'])}** — {r['second']['pct']:.1f}%")
                hm_key = f"hm_{r['id']}"
                if st.toggle(tr("heatmap_toggle"), key=hm_key):
                    ds_for_cam = st.session_state.get("results_ds_choice", "dataset_1")
                    model_for_cam, class_names_for_cam, _ = load_model(ds_for_cam)
                    class_idx = class_names_for_cam.index(r["label"])
                    cam = compute_gradcam(model_for_cam, ds_for_cam, r["image"], class_idx)
                    st.image(overlay_heatmap(r["image"], cam), use_container_width=True)
                st.info(f"{tr('advice_title')}\n\n{advisory_for(r['label'])}")
                if not st.button(tr("remove"), key=f"rm_{r['id']}"):
                    keep.append(r)
        if len(keep) != len(st.session_state.results):
            st.session_state.results = keep
            st.rerun()

# ---------------- Download tab ----------------
with tab_download:
    if not st.session_state.results:
        st.info(tr("no_report_yet"))
    else:
        st.markdown(f"#### {tr('download_title')}")
        p = st.session_state.patient or {}
        st.write(f"**{tr('f_name')[:-2]}:** {p.get('name', '-')}  ·  **{tr('f_doctor')[:-2]}:** {p.get('doctor', '-')}")
        for i, r in enumerate(st.session_state.results, start=1):
            st.write(f"{i}. **{pretty_label(r['label'])}** — {r['pct']:.1f}% ({r['model']})")
        pdf_bytes = build_pdf_report(p, st.session_state.results, [r["image"] for r in st.session_state.results])
        st.download_button(
            tr("download_btn"), data=pdf_bytes,
            file_name=f"naildx_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf", type="primary", use_container_width=True,
        )

# ---------------- Archive tab ----------------
with tab_archive:
    query = st.text_input(tr("archive_search"))
    records = st.session_state.archive
    if query:
        records = [r for r in records if query.strip().lower() in r["name"].lower()]

    if not records:
        st.info(tr("archive_empty"))
    else:
        for rec in records:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 1])
                initial = (rec["name"][:1] or "?").upper()
                c1.markdown(
                    f"""<div class="ndx-archive-row">
                        <span class="ndx-avatar">{initial}</span>
                        <div><div class="ndx-archive-name">{rec['name']}</div>
                        <div class="ndx-archive-meta">{rec['meta']} · {rec['date']}</div></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                c2.markdown(f'<span class="ndx-result-pill">{rec["result_summary"]}</span>', unsafe_allow_html=True)
                if c3.button(tr("archive_remove"), key=f"arch_rm_{rec['id']}"):
                    st.session_state.archive = [r for r in st.session_state.archive if r["id"] != rec["id"]]
                    save_archive(st.session_state.archive)
                    st.rerun()
                with st.expander(tr("archive_view")):
                    for res in rec["results"]:
                        st.write(f"- {pretty_label(res['label'])} — {res['pct']:.1f}% ({res['model']})")
                    st.caption(f"Doctor: {rec.get('doctor', '-')}")

# ---------------- About tab ----------------
with tab_about:
    st.markdown(
        f"""
        <div class="ndx-card">
          <div class="ndx-card-title">🧠 {tr('about_heading')}</div>
          <hr>
          <div class="ndx-disease-item">{tr('about_body')}</div>
        </div>
        <div class="ndx-card">
          <div class="ndx-card-title">📊 {tr('dataset_heading')}</div>
          <hr>
          <div class="ndx-disease-item">{tr('dataset_body')}</div>
        </div>
        <div class="ndx-card">
          <div class="ndx-card-title">🔗 {tr('source_heading')}</div>
          <hr>
          <div class="ndx-disease-item">{tr('source_body')}</div>
          <div class="ndx-disease-item" style="margin-top:10px">
            <span class="ndx-disease-dot">●</span>
            <a href="https://www.who.int/news-room/fact-sheets" target="_blank" style="color:{ACCENT_START};font-weight:700;">{tr('source_link_who')}</a>
          </div>
          <div class="ndx-disease-item">
            <span class="ndx-disease-dot">●</span>
            <a href="https://dermnetnz.org/" target="_blank" style="color:{ACCENT_START};font-weight:700;">{tr('source_link_dermnet')}</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
