from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageStat

from utils.consult import build_consult_messages
from utils.logger import load_local_feedback, save_feedback
from utils.model import LABELS, generate_cam, get_model_fingerprint, predict_scores
from utils.multimodal import build_ai_consult_multimodal_context
from utils.paths import LABEL_CSV, MODELS_DIR, TRAIN_IMAGES_DIR
try:
    from utils import rag as _rag_module
    _RAG_AVAILABLE = True
except Exception:
    _rag_module = None
    _RAG_AVAILABLE = False


def load_project_env():
    """Load simple KEY=VALUE pairs from a local .env file without extra dependencies."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


def local_image_data_uri(relative_path: str) -> str:
    asset_path = Path(__file__).resolve().parent / relative_path
    if not asset_path.exists():
        return ""
    ext = asset_path.suffix.lower().lstrip(".") or "png"
    mime_ext = "jpeg" if ext in {"jpg", "jpeg"} else ext
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:image/{mime_ext};base64,{encoded}"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com"))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
COPYRIGHT_TEXT = "Developed by Tian Sukai"
DISEASE_NAMES = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other disease",
}

DISEASE_DETAILS = {
    "N": {
        "title": "Normal",
        "symptoms": "Often asymptomatic; screening confirms no obvious abnormal label.",
        "risk": "Low-risk reference state, but clinical history still matters.",
        "fundus": "No obvious disease-specific lesions on fundus review.",
        "tests": "Routine visual acuity and periodic fundus screening.",
    },
    "D": {
        "title": "Diabetic Retinopathy",
        "symptoms": "Blurred vision, floaters, reduced central vision, or no symptoms early.",
        "risk": "Diabetes duration, poor glycemic control, hypertension, renal disease.",
        "fundus": "Microaneurysms, hemorrhages, hard exudates, cotton wool spots, neovascular change.",
        "tests": "Dilated fundus exam, OCT, fluorescein angiography when indicated.",
    },
    "G": {
        "title": "Glaucoma",
        "symptoms": "Peripheral field loss, halos, eye pain in acute cases, often silent early.",
        "risk": "Age, high intraocular pressure, family history, high myopia.",
        "fundus": "Optic disc cupping, rim thinning, nerve fiber layer defects.",
        "tests": "IOP, OCT RNFL, visual field testing, gonioscopy.",
    },
    "C": {
        "title": "Cataract",
        "symptoms": "Progressive blur, glare, halos, poor night vision, contrast loss.",
        "risk": "Age, diabetes, steroid use, trauma, UV exposure.",
        "fundus": "Media opacity may reduce image clarity and retinal visibility.",
        "tests": "Slit-lamp exam, visual acuity, glare testing when needed.",
    },
    "A": {
        "title": "Age-related Macular Degeneration",
        "symptoms": "Central distortion, central blur, reading difficulty, scotoma.",
        "risk": "Age, smoking, family history, cardiovascular risk.",
        "fundus": "Drusen, pigment change, geographic atrophy, macular hemorrhage or exudation.",
        "tests": "Macular OCT, fundus autofluorescence, angiography for neovascular suspicion.",
    },
    "H": {
        "title": "Hypertensive Retinopathy",
        "symptoms": "Often asymptomatic; severe cases may cause blur or headache-associated visual symptoms.",
        "risk": "Longstanding or severe hypertension, vascular disease.",
        "fundus": "Arteriolar narrowing, AV nicking, hemorrhage, cotton wool spots, disc edema.",
        "tests": "Blood pressure assessment, dilated fundus exam, systemic vascular workup.",
    },
    "M": {
        "title": "Pathologic Myopia",
        "symptoms": "Reduced vision, distortion, floaters, flashes if retinal complication occurs.",
        "risk": "High axial myopia, long axial length, family history.",
        "fundus": "Tessellated fundus, peripapillary atrophy, lacquer cracks, posterior staphyloma.",
        "tests": "Dilated peripheral exam, OCT, axial length, retinal tear evaluation if symptomatic.",
    },
    "O": {
        "title": "Other disease",
        "symptoms": "Depends on the underlying abnormality.",
        "risk": "Case-specific; requires clinician review.",
        "fundus": "Findings outside the seven primary ODIR categories.",
        "tests": "Directed ophthalmic examination based on image findings and symptoms.",
    },
}

DISEASE_ACTIONS = {
    "N": "Confirm image quality and continue routine screening if the patient has no red-flag symptoms.",
    "D": "Check diabetes duration, HbA1c history, macular OCT, and refer according to retinopathy severity.",
    "G": "Review optic disc, IOP, OCT RNFL, and visual field testing. Escalate urgently if acute symptoms exist.",
    "C": "Correlate with visual acuity, glare symptoms, and slit-lamp findings before discussing cataract pathway.",
    "A": "Ask about central distortion and reading difficulty. Consider macular OCT and urgent review for wet AMD signs.",
    "H": "Measure blood pressure and look for systemic vascular risk. Severe findings require systemic assessment.",
    "M": "Assess peripheral retina and symptoms such as flashes or floaters. Consider OCT for macular complications.",
    "O": "Use clinician review to identify the likely abnormality and decide targeted follow-up testing.",
}

GRAPH_NODES = [
    {"id": "RetinaScope", "label": "RetinaScope", "kind": "core", "x": 520, "y": 330, "r": 18},
    {"id": "Fundus Image", "label": "Fundus Image", "kind": "source", "x": 420, "y": 300, "r": 11},
    {"id": "Risk Factors", "label": "Risk Factors", "kind": "domain", "x": 330, "y": 210, "r": 10},
    {"id": "Symptoms", "label": "Symptoms", "kind": "domain", "x": 345, "y": 430, "r": 10},
    {"id": "Clinical Tests", "label": "Clinical Tests", "kind": "domain", "x": 690, "y": 210, "r": 10},
    {"id": "ODIR Labels", "label": "ODIR Labels", "kind": "domain", "x": 700, "y": 430, "r": 10},
    {"id": "Normal", "label": "Normal", "kind": "disease", "label_code": "N", "x": 865, "y": 355, "r": 12},
    {"id": "Diabetic Retinopathy", "label": "Diabetic Retinopathy", "kind": "disease", "label_code": "D", "x": 790, "y": 505, "r": 13},
    {"id": "Glaucoma", "label": "Glaucoma", "kind": "disease", "label_code": "G", "x": 790, "y": 155, "r": 13},
    {"id": "Cataract", "label": "Cataract", "kind": "disease", "label_code": "C", "x": 585, "y": 95, "r": 12},
    {"id": "AMD", "label": "AMD", "kind": "disease", "label_code": "A", "x": 935, "y": 230, "r": 12},
    {"id": "Hypertensive Retinopathy", "label": "Hypertensive Retinopathy", "kind": "disease", "label_code": "H", "x": 590, "y": 565, "r": 13},
    {"id": "Pathologic Myopia", "label": "Pathologic Myopia", "kind": "disease", "label_code": "M", "x": 410, "y": 570, "r": 12},
    {"id": "Other Disease", "label": "Other Disease", "kind": "disease", "label_code": "O", "x": 925, "y": 455, "r": 11},
    {"id": "Diabetes", "label": "Diabetes", "kind": "risk", "x": 170, "y": 120, "r": 7},
    {"id": "High BP", "label": "High BP", "kind": "risk", "x": 205, "y": 205, "r": 7},
    {"id": "Age", "label": "Age", "kind": "risk", "x": 255, "y": 120, "r": 7},
    {"id": "Smoking", "label": "Smoking", "kind": "risk", "x": 255, "y": 285, "r": 7},
    {"id": "Family History", "label": "Family History", "kind": "risk", "x": 140, "y": 275, "r": 7},
    {"id": "High Myopia", "label": "High Myopia", "kind": "risk", "x": 265, "y": 515, "r": 7},
    {"id": "Blurred Vision", "label": "Blurred Vision", "kind": "symptom", "x": 160, "y": 395, "r": 7},
    {"id": "Floaters", "label": "Floaters", "kind": "symptom", "x": 205, "y": 480, "r": 7},
    {"id": "Field Loss", "label": "Field Loss", "kind": "symptom", "x": 250, "y": 365, "r": 7},
    {"id": "Central Distortion", "label": "Central Distortion", "kind": "symptom", "x": 150, "y": 560, "r": 7},
    {"id": "Glare", "label": "Glare", "kind": "symptom", "x": 295, "y": 440, "r": 7},
    {"id": "OCT", "label": "OCT", "kind": "test", "x": 835, "y": 95, "r": 7},
    {"id": "IOP", "label": "IOP", "kind": "test", "x": 650, "y": 85, "r": 7},
    {"id": "Visual Field", "label": "Visual Field", "kind": "test", "x": 735, "y": 80, "r": 7},
    {"id": "Slit Lamp", "label": "Slit Lamp", "kind": "test", "x": 535, "y": 165, "r": 7},
    {"id": "Fundus Exam", "label": "Fundus Exam", "kind": "test", "x": 935, "y": 125, "r": 7},
    {"id": "Macular OCT", "label": "Macular OCT", "kind": "test", "x": 990, "y": 305, "r": 7},
    {"id": "RNFL OCT", "label": "RNFL OCT", "kind": "test", "x": 700, "y": 145, "r": 7},
    {"id": "Amsler Grid", "label": "Amsler Grid", "kind": "test", "x": 1000, "y": 220, "r": 7},
]

GRAPH_EDGES = [
    ("RetinaScope", "Fundus Image"), ("RetinaScope", "Risk Factors"), ("RetinaScope", "Symptoms"),
    ("RetinaScope", "Clinical Tests"), ("RetinaScope", "ODIR Labels"),
    ("ODIR Labels", "Normal"), ("ODIR Labels", "Diabetic Retinopathy"), ("ODIR Labels", "Glaucoma"),
    ("ODIR Labels", "Cataract"), ("ODIR Labels", "AMD"), ("ODIR Labels", "Hypertensive Retinopathy"),
    ("ODIR Labels", "Pathologic Myopia"), ("ODIR Labels", "Other Disease"),
    ("Diabetes", "Diabetic Retinopathy"), ("High BP", "Hypertensive Retinopathy"), ("High BP", "Diabetic Retinopathy"),
    ("Age", "Cataract"), ("Age", "AMD"), ("Age", "Glaucoma"), ("Smoking", "AMD"),
    ("Family History", "Glaucoma"), ("High Myopia", "Pathologic Myopia"), ("High Myopia", "Glaucoma"),
    ("Blurred Vision", "Cataract"), ("Blurred Vision", "Diabetic Retinopathy"), ("Floaters", "Pathologic Myopia"),
    ("Field Loss", "Glaucoma"), ("Central Distortion", "AMD"), ("Glare", "Cataract"),
    ("OCT", "Diabetic Retinopathy"), ("OCT", "AMD"), ("IOP", "Glaucoma"), ("Visual Field", "Glaucoma"),
    ("Slit Lamp", "Cataract"), ("Fundus Exam", "Hypertensive Retinopathy"), ("Fundus Exam", "Diabetic Retinopathy"),
    ("Macular OCT", "AMD"), ("Macular OCT", "Diabetic Retinopathy"), ("RNFL OCT", "Glaucoma"), ("Amsler Grid", "AMD"),
]

GRAPH_NODE_LABELS = {node["id"]: node["label"] for node in GRAPH_NODES}
GRAPH_DISEASE_IDS = {node["label_code"]: node["id"] for node in GRAPH_NODES if node.get("label_code")}
DISEASE_QUERY_TERMS = {
    "N": "正常眼底 无明显异常 常规筛查",
    "D": "糖尿病视网膜病变 糖网 黄斑水肿 飞蚊 眼底出血",
    "G": "青光眼 眼压升高 视野缺损 RNFL OCT",
    "C": "白内障 晶状体混浊 眩光 视物模糊 裂隙灯",
    "A": "AMD 年龄相关性黄斑变性 黄斑 视物变形 Amsler OCT",
    "H": "高血压视网膜病变 高血压 眼底出血 视盘水肿",
    "M": "病理性近视 高度近视 飞蚊 闪光 CNV OCT",
    "O": "其他眼病 眼底异常 转诊 复查",
}


def graph_neighbors(label: str) -> list[str]:
    disease_id = GRAPH_DISEASE_IDS.get(label)
    if not disease_id:
        return []
    neighbors = []
    for left, right in GRAPH_EDGES:
        if left == disease_id and right != "ODIR Labels":
            neighbors.append(GRAPH_NODE_LABELS.get(right, right))
        elif right == disease_id and left != "ODIR Labels":
            neighbors.append(GRAPH_NODE_LABELS.get(left, left))
    return neighbors


def build_case_rag_query(symptoms="", additional_info="", patient_history="", patient_age=None, patient_sex=None) -> str:
    active_predictions = st.session_state.get("active_predictions", {})
    active_probabilities = st.session_state.get("active_probabilities", {})
    active_labels = [label for label, value in active_predictions.items() if value == 1]
    if not active_labels and active_probabilities:
        active_labels = [max(active_probabilities, key=active_probabilities.get)]

    query_parts = [symptoms, additional_info, patient_history]
    patient_meta = st.session_state.get("active_patient_meta", {})
    if patient_meta:
        query_parts.append(f"年龄 {patient_meta.get('age', '')} 性别 {patient_meta.get('sex', '')}")
    query_parts.append(patient_meta.get("left_keywords", ""))
    query_parts.append(patient_meta.get("right_keywords", ""))
    if patient_age is not None or patient_sex is not None:
        query_parts.append(f"age {patient_age or ''} sex {patient_sex or ''}")
    quality_summary = st.session_state.get("active_quality_summary", "")
    if quality_summary:
        query_parts.append(f"图像质量 {quality_summary}")
    for label in active_labels:
        query_parts.append(f"{label} {DISEASE_NAMES[label]} {DISEASE_QUERY_TERMS.get(label, '')}")
    query_parts.append("转诊标准 红旗症状 建议检查")
    return " ".join(str(part) for part in query_parts if str(part).strip())


def render_rag_evidence(query: str, limit: int = 3):
    if not _RAG_AVAILABLE or not query.strip():
        return []
    try:
        evidence = _rag_module.explain_query(query, n_results=limit)
    except Exception as exc:
        st.caption(f"本地知识库检索暂不可用：{exc}")
        return []
    results = evidence.get("results", [])
    if not results:
        st.caption("本地知识库未检索到匹配片段。")
        return []
    st.caption(
        "本地知识库命中："
        + ", ".join(evidence.get("sources", []))
        + "；关键词："
        + "、".join(evidence.get("top_terms", [])[:8])
    )
    for result in results[:limit]:
        preview = str(result["text"]).replace("\n", " ")[:180]
        st.markdown(
            f"- `{result['source']}` score={result['score']}：{preview}..."
        )
    return results


def merge_eye_images(left_img: Image.Image, right_img: Image.Image, enhance: bool = True) -> Image.Image:
    """Merge left and right eye images into one side-by-side RGB image."""
    left_img = left_img.convert("RGB")
    right_img = right_img.convert("RGB")
    if enhance:
        left_img = ImageEnhance.Brightness(left_img).enhance(1.2)
        right_img = ImageEnhance.Brightness(right_img).enhance(1.2)

    def crop_black_border(image: Image.Image) -> Image.Image:
        bg = Image.new(image.mode, image.size, (0, 0, 0))
        bbox = ImageChops.difference(image, bg).getbbox()
        return image.crop(bbox) if bbox else image

    if left_img.size != right_img.size:
        left_img = crop_black_border(left_img)
        right_img = crop_black_border(right_img).resize(left_img.size)

    merged_img = Image.new("RGB", (left_img.width + right_img.width, max(left_img.height, right_img.height)))
    merged_img.paste(left_img, (0, 0))
    merged_img.paste(right_img, (left_img.width, 0))
    return merged_img


def image_merge(left_path: Path, right_path: Path, enhance: bool = True) -> Image.Image:
    """Load left/right image files and merge them into one side-by-side RGB image."""
    with Image.open(left_path) as left_img, Image.open(right_path) as right_img:
        return merge_eye_images(left_img, right_img, enhance=enhance)


def highlight_pred(col, primary_color):
    return [f"background-color: {primary_color}; color: white; font-weight: 700" if v == 1 else "" for v in col]


def create_avatar(background_color: str, role: str, size: int = 60) -> Image.Image:
    """Create a small in-memory avatar for chat messages."""
    img = Image.new("RGB", (size, size), color=background_color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([(2, 2), (size - 2, size - 2)], outline="white", width=3)

    head_radius = size // 8
    head_y = size // 3
    draw.ellipse(
        [(size // 2 - head_radius, head_y - head_radius), (size // 2 + head_radius, head_y + head_radius)],
        fill="white",
    )
    if role == "doctor":
        hat_height = size // 10
        draw.polygon(
            [
                (size // 2 - head_radius, head_y - head_radius),
                (size // 2, head_y - head_radius - hat_height),
                (size // 2 + head_radius, head_y - head_radius),
            ],
            fill="white",
        )
        cross_size = size // 12
        cross_y = size // 2
        draw.line([(size // 2 - cross_size, cross_y), (size // 2 + cross_size, cross_y)], fill=background_color, width=2)
        draw.line([(size // 2, cross_y - cross_size), (size // 2, cross_y + cross_size)], fill=background_color, width=2)

    body_top = head_y + head_radius
    body_bottom = size * 2 // 3
    draw.line([(size // 2, body_top), (size // 2, body_bottom)], fill="white", width=3)
    draw.ellipse([(size // 4, size - size // 6), (size - size // 4, size - size // 12)], fill="white", width=2)
    return img


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def prediction_cache_key(name, image: Image.Image, device: str) -> str:
    image = image.convert("RGB")
    hasher = hashlib.sha256()
    hasher.update(str(name).encode("utf-8", errors="ignore"))
    hasher.update(str(device).encode("utf-8", errors="ignore"))
    hasher.update(get_model_fingerprint(MODELS_DIR).encode("ascii"))
    hasher.update(f"{image.mode}:{image.size}".encode("utf-8"))
    hasher.update(image.tobytes())
    return f"prediction_cache_{hasher.hexdigest()}"


def show_image(image, caption=None, width=None, fill_width=False):
    """Render images across Streamlit versions."""
    if width is not None:
        st.image(image, caption=caption, width=width)
    elif fill_width:
        st.image(image, caption=caption, width="stretch")
    else:
        st.image(image, caption=caption)


def render_cam_image(image, caption=None, max_width=460, max_height=320):
    """Render GradCAM output at a stable, inspection-friendly size."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image = image.convert("RGB")
    display = image.copy()
    display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    st.image(display, caption=caption, width=min(display.width, max_width))


def render_diagnostic_image(image, caption=None, max_width=520, max_height=320):
    """Render fundus images without stretching them to the full column width."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    caption_html = f'<div class="cam-caption">{html.escape(caption)}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="diagnostic-image-wrap">
            <img src="data:image/png;base64,{image_b64}" alt="{html.escape(caption or 'Fundus image')}"
                 style="width: 100%; max-width: {max_width}px; max-height: {max_height}px; object-fit: contain;">
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warm_table(df: pd.DataFrame, max_rows: int | None = None):
    """Render read-only tables without inheriting Streamlit's dark dataframe canvas."""
    display_df = df.head(max_rows).copy() if max_rows else df.copy()
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in display_df.columns)
    rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"""
        <div class="warm-table-wrap">
            <table class="warm-table">
                <thead><tr>{headers}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def deepseek_chat_completions_url() -> str:
    """Build the OpenAI-compatible DeepSeek chat completions endpoint."""
    base_url = DEEPSEEK_API_BASE_URL.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/anthropic"):
        base_url = base_url[: -len("/anthropic")]
    return f"{base_url}/chat/completions"


def call_deepseek_api_stream(messages: list[dict[str, str]], api_key: str, message_placeholder):
    """Call DeepSeek official chat completion API and stream the response into Streamlit."""
    if not api_key:
        raise RuntimeError("DeepSeek API key is not configured")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        "max_tokens": 1600,
    }

    try:
        response = requests.post(
            deepseek_chat_completions_url(),
            headers=headers,
            json=payload,
            stream=True,
            timeout=(10, 60),
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data_json = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = data_json.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_response += content
                message_placeholder.markdown(full_response + "▌")
        if not full_response.strip():
            raise RuntimeError("DeepSeek returned an empty response")
        message_placeholder.markdown(full_response)
        return full_response
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("DeepSeek request timed out") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("DeepSeek request failed") from exc


@st.cache_data
def load_metadata() -> pd.DataFrame:
    if LABEL_CSV.exists():
        return pd.read_csv(LABEL_CSV).set_index("ID")
    return pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Patient Age": [45, 60, 30],
            "Patient Sex": ["Male", "Female", "Male"],
            "N": [1, 0, 0],
            "D": [0, 1, 0],
            "G": [0, 0, 1],
            "C": [0, 0, 0],
            "A": [0, 0, 0],
            "H": [0, 0, 0],
            "M": [0, 0, 0],
            "O": [0, 0, 0],
            "Left-Diagnostic Keywords": ["Normal", "Diabetes", "Glaucoma"],
            "Right-Diagnostic Keywords": ["Normal", "Diabetes", "Glaucoma"],
        }
    ).set_index("ID")


def inject_theme(dark_mode: bool):
    colors = {
        "bg": "#f6faf8",
        "surface": "#ffffff",
        "surface_raised": "#ffffff",
        "text": "#102a27",
        "muted": "#667874",
        "primary": "#006b4e",
        "secondary": "#16815f",
        "accent": "#e8f6ef",
        "sidebar": "#003f32",
        "form": "#ffffff",
        "border": "#dbe7e2",
        "user": "#eaf7f1",
        "assistant": "#f4f8f6",
        "warning": "#b7791f",
        "danger": "#d92d20",
    }

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap');
        html, body, .stApp, input, select, textarea {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        }}
        .stMarkdown, .stText, .stAlert, .stInfo, .stSuccess, .stWarning, .stError,
        [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stCaptionContainer"], [data-testid="stHeadingWithActionElements"],
        [data-testid="stExpander"] summary,
        [data-baseweb="tab"] span, [data-baseweb="typo-label"],
        [data-baseweb="form-control"] label, [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        }}
        .stApp {{
            background:
                linear-gradient(180deg, #fbfdfc 0%, #f6faf8 48%, #eef7f2 100%);
            color: {colors["text"]};
        }}
        .block-container {{
            padding-top: 0.85rem;
            max-width: 1680px;
            padding-left: 2.25rem !important;
            padding-right: 2.25rem !important;
        }}
        header[data-testid="stHeader"] {{
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid {colors["border"]};
            height: 42px;
        }}
        header[data-testid="stHeader"]::before,
        header[data-testid="stHeader"]::after {{
            background: {colors["surface"]} !important;
        }}
        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stDeployButton"] {{
            background: {colors["surface"]} !important;
            color: {colors["muted"]} !important;
        }}
        div[data-testid="stDecoration"] {{
            display: none;
        }}
        *:focus {{
            outline: none !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, #004b3a 0%, #003f32 58%, #002b23 100%);
            border-right: 1px solid rgba(255,255,255,0.10);
            box-shadow: 18px 0 42px rgba(0, 63, 50, 0.13);
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
            color: rgba(255,255,255,0.92) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            margin-bottom: 10px;
        }}
        .sidebar-brand {{
            padding: 8px 2px 18px;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.16);
        }}
        .sidebar-brand-mark {{
            width: 42px;
            height: 42px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.28);
            color: #ffffff;
            font-weight: 900;
            vertical-align: middle;
        }}
        .sidebar-brand-text {{
            display: inline-block;
            vertical-align: middle;
        }}
        .sidebar-brand-title {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 850;
            line-height: 1.05;
        }}
        .sidebar-brand-subtitle {{
            color: rgba(255,255,255,0.72);
            font-size: 12px;
            margin-top: 3px;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {colors["border"]};
            padding: 4px 0 0;
            background: rgba(255,255,255,0.82);
        }}
        button[data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            background: transparent;
            border: 0;
            color: {colors["muted"]};
            padding: 9px 15px;
            font-weight: 650;
            font-size: 13px;
            letter-spacing: 0;
            transition: color 0.18s ease, background 0.18s ease;
            white-space: nowrap;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: #ffffff;
            border-bottom: 3px solid {colors["primary"]};
            color: {colors["text"]};
            font-weight: 800;
            box-shadow: 0 -1px 16px rgba(0,107,78,0.08);
        }}
        button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
            color: {colors["primary"]};
            background: rgba(0,107,78,0.07);
        }}
        input::placeholder, textarea::placeholder {{
            color: #7b8985 !important;
            opacity: 1 !important;
        }}
        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stMarkdownContainer"] label,
        [data-baseweb="form-control"] label {{
            color: {colors["text"]} !important;
            opacity: 1 !important;
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        section[data-testid="stSidebar"] [data-baseweb="form-control"] label {{
            color: rgba(255,255,255,0.92) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: rgba(255,255,255,0.68) !important;
        }}
        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] [role="button"],
        div[data-baseweb="base-input"],
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {{
            background: #ffffff !important;
            color: {colors["text"]} !important;
            border-color: {colors["border"]} !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 0 rgba(16, 42, 39, 0.03) !important;
        }}
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: {colors["primary"]} !important;
        }}
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"] *,
        div[data-baseweb="select"] *,
        [data-testid="stTextInput"] *,
        [data-testid="stTextArea"] *,
        [data-testid="stNumberInput"] * {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: #f6faf8 !important;
            border: 1px dashed #b9cec6 !important;
            border-radius: 8px !important;
            color: {colors["text"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: {colors["surface"]} !important;
            border: 1px solid {colors["border"]} !important;
            color: {colors["primary"]} !important;
        }}
        [data-testid="stFileUploaderDropzoneInput"],
        [data-testid="stFileUploaderDropzone"] input[type="file"] {{
            width: 0.1px !important;
            height: 0.1px !important;
            opacity: 0 !important;
            overflow: hidden !important;
            position: absolute !important;
            z-index: -1 !important;
            pointer-events: none !important;
        }}
        [data-testid="stNumberInput"] button span,
        [data-testid="stNumberInput"] [data-baseweb="button"] > span:not(:empty):not([data-testid]) {{
            display: none !important;
        }}
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stDataFrame"] div,
        [data-testid="stTable"] div {{
            background: {colors["surface"]} !important;
            color: {colors["text"]} !important;
        }}
        .warm-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: {colors["surface"]};
            margin: 10px 0 18px;
        }}
        .warm-table {{
            width: 100%;
            border-collapse: collapse;
            color: {colors["text"]};
            font-size: 14px;
        }}
        .warm-table th {{
            background: #f1f7f4;
            color: {colors["text"]};
            text-align: left;
            font-weight: 700;
            padding: 11px 12px;
            border-bottom: 1px solid {colors["border"]};
            white-space: nowrap;
        }}
        .warm-table td {{
            background: {colors["surface"]};
            color: {colors["text"]};
            padding: 10px 12px;
            border-bottom: 1px solid #e7efeb;
            vertical-align: top;
            white-space: nowrap;
        }}
        .warm-table tr:nth-child(even) td {{
            background: #f8fbfa;
        }}
        .warm-table tr:last-child td {{
            border-bottom: 0;
        }}
        .clinical-hero {{
            padding: 22px 24px;
            border-radius: 8px;
            margin-bottom: 16px;
            color: {colors["text"]};
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(236,248,242,0.98) 58%, rgba(250,253,251,0.98) 100%);
            border: 1px solid {colors["border"]};
            box-shadow: 0 12px 30px rgba(0, 63, 50, 0.07);
        }}
        .clinical-hero h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 850;
            letter-spacing: 0;
        }}
        .clinical-hero p {{
            margin: 4px 0 0;
            max-width: 760px;
            color: {colors["muted"]};
            font-size: 14px;
        }}
        .hero-layout {{
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(360px, .9fr);
            gap: 22px;
            align-items: center;
        }}
        .hero-kicker {{
            color: {colors["primary"]};
            font-size: 12px;
            font-weight: 850;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .hero-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }}
        .hero-status-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .hero-status-grid .stat-card {{
            margin: 0;
            min-height: 104px;
        }}
        .hero-status-grid .stat-value {{
            font-size: 22px !important;
        }}
        .clinical-hero .hero-mark {{
            display: none;
        }}
        .stat-card, .clinical-card, .doctor-feedback-form, .error-case-card {{
            background: {colors["form"]};
            border: 1px solid {colors["border"]};
            border-left: 0;
            border-radius: 8px;
            padding: 18px;
            margin: 12px 0;
            box-shadow: 0 8px 22px rgba(0, 63, 50, 0.045);
        }}
        .stat-card {{
            text-align: left;
            border-left-width: 1px;
            min-height: 132px;
        }}
        .stat-value {{
            color: {colors["primary"]};
            font-size: 26px;
            font-weight: 800;
        }}
        .stat-label {{
            color: {colors["text"]};
            font-size: 13px;
            opacity: 0.78;
        }}
        .metric-hint {{
            color: {colors["muted"]};
            font-size: 12px;
            margin-top: 5px;
        }}
        .overview-panel {{
            display: grid;
            grid-template-columns: minmax(420px, 1fr) minmax(0, 1.15fr);
            gap: 18px;
            align-items: stretch;
            margin: 4px 0 8px;
        }}
        .overview-copy,
        .flow-item {{
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: {colors["surface"]};
            box-shadow: 0 6px 18px rgba(0, 63, 50, 0.04);
        }}
        .overview-copy {{
            padding: 20px;
        }}
        .overview-copy h2 {{
            margin: 4px 0 8px;
            font-size: 22px;
            line-height: 1.25;
            letter-spacing: 0;
            color: {colors["text"]};
        }}
        .overview-copy p {{
            margin: 0;
            color: {colors["muted"]};
            line-height: 1.72;
            font-size: 14px;
        }}
        .section-kicker {{
            color: {colors["primary"]};
            font-size: 12px;
            font-weight: 850;
            letter-spacing: .11em;
            text-transform: uppercase;
        }}
        .overview-flow {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }}
        .flow-item {{
            padding: 16px;
            min-height: 126px;
        }}
        .flow-item strong {{
            display: block;
            color: {colors["text"]};
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .flow-item span {{
            color: {colors["muted"]};
            font-size: 12.5px;
            line-height: 1.65;
        }}
        .overview-carousel {{
            position: relative;
            min-height: 390px;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            overflow: hidden;
            background: {colors["surface"]};
            box-shadow: 0 10px 26px rgba(0, 63, 50, 0.06);
        }}
        .overview-slide {{
            position: absolute;
            inset: 0;
            opacity: 0;
            animation: overviewSlide 18s infinite;
        }}
        .overview-slide:nth-child(2) {{ animation-delay: 6s; }}
        .overview-slide:nth-child(3) {{ animation-delay: 12s; }}
        @keyframes overviewSlide {{
            0%, 28% {{ opacity: 1; transform: scale(1); }}
            34%, 100% {{ opacity: 0; transform: scale(1.018); }}
        }}
        .overview-slide img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        .overview-slide::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(20,42,38,0.02), rgba(20,42,38,0.56));
        }}
        .overview-slide-caption {{
            position: absolute;
            left: 20px;
            right: 20px;
            bottom: 18px;
            z-index: 2;
            color: #ffffff;
        }}
        .overview-slide-caption strong {{
            display: block;
            font-size: 19px;
            margin-bottom: 4px;
        }}
        .overview-slide-caption span {{
            font-size: 12.5px;
            opacity: .9;
        }}
        .overview-source {{
            color: {colors["muted"]};
            font-size: 11px;
            margin-top: 6px;
            opacity: .72;
        }}
        .page-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            padding: 18px 20px;
            margin: 4px 0 14px;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: rgba(255,255,255,0.94);
            box-shadow: 0 8px 22px rgba(0, 63, 50, 0.045);
        }}
        .page-header h2 {{
            margin: 0;
            color: {colors["text"]};
            font-size: 25px;
            font-weight: 820;
            letter-spacing: 0;
        }}
        .page-header p {{
            margin: 7px 0 0;
            max-width: 780px;
            color: {colors["muted"]};
            font-size: 13.5px;
            line-height: 1.65;
        }}
        .page-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-end;
        }}
        .soft-panel {{
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: rgba(255,255,255,0.88);
            padding: 16px;
            box-shadow: 0 6px 18px rgba(0, 63, 50, 0.04);
            margin: 10px 0 14px;
        }}
        .filter-panel {{
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: #ffffff;
            padding: 14px 14px 4px;
            margin: 10px 0 14px;
        }}
        .meta-strip {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 12px;
        }}
        .meta-pill {{
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: #ffffff;
            padding: 11px 12px;
        }}
        .meta-pill b {{
            display: block;
            color: {colors["text"]};
            font-size: 13px;
            margin-bottom: 4px;
        }}
        .meta-pill span {{
            color: {colors["muted"]};
            font-size: 12px;
            line-height: 1.45;
        }}
        .empty-state {{
            border: 1px dashed #b9cec6;
            border-radius: 8px;
            background: #ffffff;
            padding: 24px;
            color: {colors["muted"]};
            text-align: center;
            margin: 14px 0;
        }}
        .quiet-copyright {{
            position: fixed;
            left: 20px;
            bottom: 12px;
            max-width: 245px;
            color: rgba(255,255,255,0.68) !important;
            font-size: 11px !important;
            opacity: .82;
            line-height: 1.35;
            pointer-events: none;
        }}
        .risk-row {{
            display: grid;
            grid-template-columns: 42px minmax(130px, 1.1fr) minmax(150px, 2fr) 58px;
            gap: 12px;
            align-items: center;
            padding: 10px 12px;
            margin-bottom: 8px;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background: {colors["surface"]};
            box-shadow: 0 2px 8px rgba(0, 63, 50, 0.025);
        }}
        .risk-row.positive {{
            border-color: #a9d8ca;
            background: #f0faf5;
        }}
        .risk-row.high {{
            border-color: #a9d8ca;
            box-shadow: inset 3px 0 0 {colors["danger"]}, 0 2px 8px rgba(0,107,78,0.08);
        }}
        .risk-row.medium {{
            border-color: #e7c36a;
        }}
        .risk-level {{
            font-size: 11px;
            color: {colors["muted"]};
            text-transform: uppercase;
        }}
        .risk-label {{
            color: {colors["secondary"]};
            font-weight: 800;
        }}
        .risk-name {{
            color: {colors["text"]};
            font-size: 13px;
        }}
        .risk-track {{
            height: 8px;
            border-radius: 999px;
            background: #e4ece8;
            overflow: hidden;
        }}
        .risk-fill {{
            height: 100%;
            border-radius: 999px;
            background: {colors["primary"]};
        }}
        .risk-value {{
            text-align: right;
            color: {colors["text"]};
            font-variant-numeric: tabular-nums;
        }}
        .quality-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 12px;
        }}
        .quality-item {{
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 9px 10px;
            background: {colors["surface"]};
        }}
        .quality-item strong {{
            display: block;
            color: {colors["text"]};
        }}
        .quality-item span {{
            color: {colors["muted"]};
            font-size: 12px;
        }}
        .quality-item.warn {{
            border-color: rgba(245, 158, 11, 0.82);
        }}
        .quality-item.fail {{
            border-color: rgba(239, 68, 68, 0.82);
        }}
        .insight-note {{
            border: 1px solid {colors["border"]};
            border-left: 4px solid {colors["primary"]};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 10px 0;
            background: #f0faf5;
            color: {colors["text"]};
        }}
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {colors["border"]};
            background: {colors["surface"]};
            color: {colors["text"]};
            min-height: 38px;
            padding: 0 20px;
            font-weight: 700;
            font-size: 13.5px;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            border-color: {colors["primary"]};
            color: {colors["primary"]};
            background: rgba(79,155,143,0.06);
            box-shadow: 0 2px 10px rgba(0,107,78,0.15);
        }}
        button[kind="primary"] {{
            background: {colors["primary"]} !important;
            border: 0 !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 650 !important;
            box-shadow: 0 4px 14px rgba(0,107,78,0.22) !important;
        }}
        .chat-container {{
            max-height: 560px;
            overflow-y: auto;
            padding: 6px;
        }}
        .message {{
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
        }}
        .user-message {{
            justify-content: flex-end;
        }}
        .avatar {{
            width: 48px;
            height: 48px;
            border-radius: 50%;
        }}
        .bubble {{
            max-width: 72%;
            padding: 12px 16px;
            border-radius: 14px;
            line-height: 1.5;
            color: {colors["text"]};
        }}
        .user-bubble {{
            background: {colors["user"]};
        }}
        .assistant-bubble {{
            background: {colors["assistant"]};
        }}
        .message-meta {{
            font-size: 12px;
            font-weight: 700;
            opacity: 0.7;
            margin-bottom: 4px;
        }}
        .cam-image-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            margin-top: 12px;
        }}
        .cam-image-wrap img {{
            width: auto;
            max-width: 100%;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid {colors["border"]};
            background: #08083a;
        }}
        .cam-caption {{
            margin-top: 8px;
            color: {colors["muted"]};
            font-size: 13px;
            text-align: center;
        }}
        .diagnostic-image-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            width: 100%;
            margin-top: 8px;
        }}
        .diagnostic-image-wrap img {{
            width: auto;
            height: auto;
            max-width: 100%;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid {colors["border"]};
            background: #08083a;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: {colors["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return colors


def inject_night_refinement(colors):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {colors["bg"]} !important;
        }}
        .clinical-hero {{
            padding: 14px 18px;
            border: 1px solid {colors["border"]};
            background: {colors["surface"]} !important;
            box-shadow: none;
        }}
        .clinical-hero h1 {{
            font-size: 22px;
        }}
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }}
        .feature-card, .pipeline-step, .knowledge-card {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 16px;
            box-shadow: none;
        }}
        .section-header {{
            margin: 22px 0 12px;
        }}
        .section-header h2 {{
            margin: 0;
            color: {colors["text"]};
            font-size: 24px;
            letter-spacing: 0;
        }}
        .section-header p {{
            margin: 6px 0 0;
            color: {colors["muted"]};
            max-width: 860px;
            line-height: 1.55;
        }}
        .handoff-card {{
            background: #f7fbf9;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 14px;
            margin: 10px 0 14px;
        }}
        .handoff-card h3 {{
            margin: 0 0 8px;
            color: {colors["text"]};
            font-size: 18px;
        }}
        .handoff-card p {{
            margin: 0;
            color: {colors["muted"]};
            line-height: 1.55;
        }}
        .quiet-note {{
            border-left: 3px solid {colors["warning"]};
            background: #fffaf0;
            color: {colors["text"]};
            padding: 12px 14px;
            border-radius: 8px;
            margin: 12px 0;
            line-height: 1.5;
        }}
        .review-summary {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            margin: 8px 0 12px;
        }}
        .review-cell {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            padding: 10px 12px;
        }}
        .review-cell span {{
            display: block;
            color: {colors["muted"]};
            font-size: 12px;
            margin-bottom: 6px;
        }}
        .review-cell strong {{
            color: {colors["text"]};
            font-size: 14px;
        }}
        .feature-card h3, .knowledge-card h3 {{
            margin: 0 0 8px;
            color: {colors["primary"]};
            font-size: 17px;
        }}
        .feature-card p, .knowledge-card p, .pipeline-step span {{
            margin: 0;
            color: {colors["muted"]};
            font-size: 13px;
            line-height: 1.55;
        }}
        .pipeline {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px;
        }}
        .pipeline-step strong {{
            display: block;
            margin-bottom: 6px;
            color: {colors["primary"]};
        }}
        .disease-chip {{
            display: inline-block;
            margin: 4px 6px 4px 0;
            padding: 5px 9px;
            border: 1px solid {colors["border"]};
            border-radius: 999px;
            background: #f0faf5;
            color: {colors["text"]};
            font-size: 12px;
        }}
        @media (max-width: 900px) {{
            .feature-grid, .pipeline, .review-summary {{
                grid-template-columns: 1fr;
            }}
            .clinical-hero h1 {{
                font-size: 30px;
            }}
        }}
        /* ── Premium card & layout enhancements ── */
        .feature-card, .knowledge-card {{
            transition: transform 0.22s ease, box-shadow 0.22s ease;
            box-shadow: 0 2px 14px rgba(0,63,50,0.07);
        }}
        .feature-card:hover, .knowledge-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 32px rgba(0,63,50,0.11);
        }}
        .pipeline-step {{
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 2px 10px rgba(0,63,50,0.06);
        }}
        .pipeline-step:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 22px rgba(0,63,50,0.10);
        }}
        .section-header h2 {{
            font-size: 28px !important;
            font-weight: 750 !important;
            letter-spacing: 0 !important;
        }}
        .section-header p {{
            font-size: 15px !important;
            line-height: 1.65 !important;
        }}
        .stat-card {{
            box-shadow: 0 2px 14px rgba(0,63,50,0.07);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 28px rgba(0,63,50,0.11);
        }}
        .stat-value {{
            font-size: 30px !important;
            font-weight: 800 !important;
            letter-spacing: 0;
        }}
        .handoff-card {{
            border-left: 4px solid {colors["primary"]} !important;
            box-shadow: 0 2px 14px rgba(0,63,50,0.07) !important;
        }}
        .handoff-card h3 {{
            font-size: 19px !important;
            font-weight: 700 !important;
        }}
        .feature-card h3 {{
            font-size: 17px !important;
            font-weight: 700 !important;
        }}
        .pipeline-step strong {{
            font-size: 13.5px !important;
            font-weight: 700 !important;
        }}
        .insight-note {{
            box-shadow: 0 2px 10px rgba(0,63,50,0.08);
        }}
        /* Make the iframe that holds carousel flush/borderless */
        iframe[title="streamlit_components_v1_html"] {{
            border: none !important;
            border-radius: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str, chips: list[str] | None = None):
    chip_html = ""
    if chips:
        chip_html = '<div class="page-chip-row">' + "".join(
            f'<span class="disease-chip">{html.escape(str(chip))}</span>' for chip in chips
        ) + "</div>"
    st.markdown(
        f"""
        <div class="page-header">
          <div>
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
          </div>
          {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def knowledge_tab():
    active_predictions = st.session_state.get("active_predictions", {})
    active_labels = {label for label, value in active_predictions.items() if value == 1}
    chips = ["图谱联动"]
    if active_labels:
        chips.append(f"当前阳性 {len(active_labels)} 项")
    render_page_header(
        "知识图谱",
        "将模型预测标签与风险因素、症状和检查项目联动，帮助解释诊断关注点。",
        chips,
    )
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    graph_col, disease_col = st.columns([1.25, 1])
    with graph_col:
        graph_search = st.text_input("图谱搜索", placeholder="输入疾病、症状、风险因素或检查项，例如：青光眼、飞蚊、OCT")
    with disease_col:
        selected_label = st.selectbox("疾病详情面板", LABELS, format_func=lambda label: f"{label} - {DISEASE_NAMES[label]}")
    st.markdown("</div>", unsafe_allow_html=True)
    render_obsidian_graph(active_labels=active_labels, selected_label=selected_label, search_query=graph_search)

    detail = DISEASE_DETAILS[selected_label]
    st.markdown(
        f"""
        <div class="knowledge-card">
            <h3>{selected_label} - {detail["title"]}</h3>
            <p><strong>症状：</strong> {detail["symptoms"]}</p>
            <p><strong>风险因素：</strong> {detail["risk"]}</p>
            <p><strong>眼底表现：</strong> {detail["fundus"]}</p>
            <p><strong>常规检查：</strong> {detail["tests"]}</p>
            <p><strong>临床建议：</strong> {DISEASE_ACTIONS[selected_label]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return


def risk_level(probability: float, positive: bool) -> tuple[str, str]:
    if positive:
        return "阳性提示", "high"
    if probability >= 0.7:
        return "需复核", "medium"
    if probability >= 0.3:
        return "观察", "medium"
    return "低风险", "low"


def assess_image_quality(image: Image.Image):
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    width, height = rgb.size

    border = max(4, min(width, height) // 40)
    border_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(border_mask)
    draw.rectangle([0, 0, width, border], fill=255)
    draw.rectangle([0, height - border, width, height], fill=255)
    draw.rectangle([0, 0, border, height], fill=255)
    draw.rectangle([width - border, 0, width, height], fill=255)
    border_pixels = np.array(gray)[np.array(border_mask) == 255]
    dark_border_ratio = float((border_pixels < 12).mean()) if border_pixels.size else 0.0

    checks = [
        {
            "name": "亮度",
            "status": "fail" if brightness < 35 else "warn" if brightness < 55 else "ok",
            "value": f"{brightness:.0f}/255",
            "note": "图像过暗" if brightness < 35 else "略暗，建议复核" if brightness < 55 else "可接受",
        },
        {
            "name": "对比度",
            "status": "warn" if contrast < 28 else "ok",
            "value": f"{contrast:.0f}",
            "note": "对比度偏低，可能影响病灶观察" if contrast < 28 else "可接受",
        },
        {
            "name": "分辨率",
            "status": "warn" if min(width, height) < 512 else "ok",
            "value": f"{width} x {height}",
            "note": "低于模型目标尺寸" if min(width, height) < 512 else "可接受",
        },
        {
            "name": "黑边",
            "status": "warn" if dark_border_ratio > 0.65 else "ok",
            "value": f"{dark_border_ratio:.0%}",
            "note": "检测到较大黑边" if dark_border_ratio > 0.65 else "可接受",
        },
    ]
    overall = "fail" if any(item["status"] == "fail" for item in checks) else "warn" if any(item["status"] == "warn" for item in checks) else "ok"
    return {"overall": overall, "checks": checks}


def render_comorbidity_alert(result_df: pd.DataFrame):
    positives = result_df[result_df["Prediction"] == 1].sort_values("Probability", ascending=False)
    if len(positives) <= 1:
        return
    labels = [row.Label for row in positives.itertuples()]
    disease_text = ", ".join([f"{row.Label} - {row.Disease}" for row in positives.itertuples()])
    suggestions = []
    if {"D", "H"}.issubset(labels):
        suggestions.append("结合血压、血糖和全身血管风险一起复核")
    if {"D", "A"}.issubset(labels) or {"D", "M"}.issubset(labels):
        suggestions.append("建议关注黄斑 OCT，因为中央视网膜并发改变可能重叠")
    if {"G", "M"}.issubset(labels):
        suggestions.append("高度近视可能影响视盘判断，青光眼线索需谨慎解释")
    if not suggestions:
        suggestions.append("多个疾病标签同时激活，建议优先人工复核")
    st.markdown(
        f"""
        <div class="insight-note">
            <strong>多标签复核提示</strong><br>
            当前阳性标签：{disease_text}。建议复核：{"；".join(suggestions)}。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_panel(result_df: pd.DataFrame):
    rows = []
    for _, row in result_df.iterrows():
        probability = float(row["Probability"])
        positive = int(row["Prediction"]) == 1
        level, css_level = risk_level(probability, positive)
        rows.append(
            f"""
            <div class="risk-row {'positive' if positive else ''} {css_level}">
                <div class="risk-label">{row['Label']}</div>
                <div class="risk-name">{row['Disease']}<div class="risk-level">{level}</div></div>
                <div class="risk-track"><div class="risk-fill" style="width: {max(0, min(probability, 1)) * 100:.1f}%"></div></div>
                <div class="risk-value">{probability:.2f}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_obsidian_graph(active_labels=None, selected_label=None, search_query=""):
    active_labels = sorted(active_labels or [])
    payload = {
        "nodes": GRAPH_NODES,
        "edges": [{"source": source, "target": target} for source, target in GRAPH_EDGES],
        "activeLabels": active_labels,
        "selectedLabel": selected_label or "",
        "searchQuery": search_query or "",
        "details": DISEASE_DETAILS,
        "names": DISEASE_NAMES,
    }
    graph_json = json.dumps(payload)
    html = f"""
    <div id="retina-graph-wrap">
      <div class="graph-shell">
        <div class="graph-topbar">
          <div>
            <strong>图谱视图</strong>
            <span>拖拽节点 · 平移画布 · 滚轮缩放</span>
          </div>
          <div class="graph-actions">
            <button id="reset-view" type="button">重置视图</button>
            <button id="settle-layout" type="button">重新布局</button>
          </div>
          <div class="graph-legend">
            <span><i class="core"></i>核心</span>
            <span><i class="disease"></i>疾病</span>
            <span><i class="risk"></i>风险因素</span>
            <span><i class="test"></i>检查项</span>
          </div>
        </div>
        <svg id="retina-graph" viewBox="0 0 1080 660" preserveAspectRatio="xMidYMid meet"></svg>
        <div id="graph-detail" class="graph-detail">
          <strong>点击选中节点</strong>
          <p>点击疾病节点查看临床信息。</p>
        </div>
      </div>
    </div>
    <style>
      #retina-graph-wrap {{ background: #f4faf7; border-radius: 14px; padding: 18px; border: 1px solid #dbe7e2; }}
      .graph-shell {{ position: relative; background: #ffffff; border: 1px solid #dbe7e2; border-radius: 12px; overflow: hidden; box-shadow: 0 14px 36px rgba(0, 63, 50, 0.08); }}
      .graph-topbar {{ height: 48px; display: flex; align-items: center; justify-content: space-between; padding: 0 16px; border-bottom: 1px solid #e7efeb; color: #102a27; font-family: Inter, Segoe UI, sans-serif; }}
      .graph-topbar strong {{ display: block; font-size: 14px; }}
      .graph-topbar span {{ color: #8a8178; font-size: 12px; }}
      .graph-actions {{ display: flex; gap: 8px; }}
      .graph-actions button {{ border: 1px solid #dbe7e2; background: #ffffff; color: #006b4e; border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer; }}
      .graph-actions button:hover {{ background: #eef7f2; }}
      .graph-legend {{ display: flex; gap: 12px; align-items: center; }}
      .graph-legend span {{ display: flex; gap: 5px; align-items: center; }}
      .graph-legend i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
      .graph-legend .core {{ background: #6ea99b; }}
      .graph-legend .disease {{ background: #8fbf89; }}
      .graph-legend .risk {{ background: #d5ad55; }}
      .graph-legend .test {{ background: #94aaa1; }}
      #retina-graph {{ width: 100%; height: 620px; display: block; background: radial-gradient(circle at 50% 45%, #ffffff, #f7fbf9); }}
      .graph-detail {{ position: absolute; right: 14px; bottom: 14px; width: 270px; max-height: 210px; overflow: auto; padding: 12px; border-radius: 10px; background: rgba(255,255,255,0.94); border: 1px solid #dbe7e2; color: #102a27; font-family: Inter, Segoe UI, sans-serif; box-shadow: 0 10px 26px rgba(0, 63, 50, 0.10); }}
      .graph-detail strong {{ font-size: 14px; }}
      .graph-detail p {{ margin: 6px 0 0; font-size: 12px; line-height: 1.45; color: #6f7d79; }}
      .node-label {{ user-select: none; pointer-events: none; }}
    </style>
    <script>
      const payload = {graph_json};
      const svg = document.getElementById("retina-graph");
      const detail = document.getElementById("graph-detail");
      const resetBtn = document.getElementById("reset-view");
      const settleBtn = document.getElementById("settle-layout");
      const ns = "http://www.w3.org/2000/svg";
      const active = new Set(payload.activeLabels || []);
      const selected = payload.selectedLabel;
      const searchQuery = (payload.searchQuery || "").trim().toLowerCase();
      const CX = 540, CY = 330, INIT_R = 210;
      const nodes = payload.nodes.map((n, i) => {{
        const angle = (i / payload.nodes.length) * Math.PI * 2 - Math.PI / 2;
        const jitter = 0.75 + Math.random() * 0.5;
        return {{...n, x: CX + INIT_R * Math.cos(angle) * jitter, y: CY + INIT_R * Math.sin(angle) * jitter, vx:0, vy:0}};
      }});
      const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
      const colors = {{ core:"#6ea99b", source:"#94aaa1", domain:"#b7c8bd", disease:"#8fbf89", risk:"#d5ad55", symptom:"#c6b690", test:"#a8beb4" }};
      const neighborMap = new Map();
      payload.edges.forEach(e => {{
        if (!neighborMap.has(e.source)) neighborMap.set(e.source, new Set());
        if (!neighborMap.has(e.target)) neighborMap.set(e.target, new Set());
        neighborMap.get(e.source).add(e.target);
        neighborMap.get(e.target).add(e.source);
      }});
      let selectedNodeId = null;
      let scale = 1;
      let tx = 0;
      let ty = 0;
      let draggingNode = null;
      let panning = null;
      let pointerMoved = false;

      function matchNode(n) {{
        if (!searchQuery) return true;
        const detail = n.label_code && payload.details[n.label_code] ? payload.details[n.label_code] : {{}};
        const haystack = [n.id, n.label, n.kind, n.label_code || "", detail.title || "", detail.symptoms || "", detail.risk || "", detail.fundus || "", detail.tests || ""].join(" ").toLowerCase();
        return haystack.includes(searchQuery);
      }}
      function make(tag, attrs={{}}) {{
        const el = document.createElementNS(ns, tag);
        Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v));
        return el;
      }}
      svg.innerHTML = "";
      const viewport = make("g", {{id:"graph-viewport"}});
      const edgeLayer = make("g", {{id:"edge-layer"}});
      const nodeLayer = make("g", {{id:"node-layer"}});
      viewport.appendChild(edgeLayer);
      viewport.appendChild(nodeLayer);
      svg.appendChild(viewport);
      const edgeElements = [];
      const nodeElements = new Map();

      payload.edges.forEach(e => {{
        const line = make("line", {{"data-source":e.source, "data-target":e.target, stroke:"#cddbd5", "stroke-width":"1", opacity:"0.55"}});
        edgeLayer.appendChild(line);
        edgeElements.push({{...e, el: line}});
      }});
      nodes.forEach(n => {{
        const isActive = n.label_code && active.has(n.label_code);
        const isSelected = n.label_code && n.label_code === selected;
        const isMatch = matchNode(n);
        const group = make("g", {{"data-id":n.id, opacity: isMatch ? "1" : "0.62", style:"cursor:grab"}});
        const halo = make("circle", {{r:n.r + (isActive || isSelected || isMatch && searchQuery ? 9 : 4), fill:isActive ? "#eef6f2" : searchQuery && isMatch ? "#e6f7ef" : "#eef4f1", opacity:isActive || isSelected || searchQuery && isMatch ? "0.88" : "0.42"}});
        const circle = make("circle", {{r:n.r, fill:isSelected ? "#006b4e" : isActive ? "#16815f" : searchQuery && isMatch ? "#006b4e" : colors[n.kind] || "#94aaa1", stroke:"#ffffff", "stroke-width": isActive || isSelected || searchQuery && isMatch ? "2.6" : "1.2"}});
        const label = make("text", {{x:n.r + 5, y:4, fill: searchQuery && isMatch ? "#102a27" : "#5f736f", "font-size": n.kind === "core" ? "12" : "9", "font-family":"Inter, Segoe UI, sans-serif", class:"node-label"}});
        label.textContent = n.label;
        group.appendChild(halo); group.appendChild(circle); group.appendChild(label);
        group.addEventListener("pointerdown", (event) => {{
          event.stopPropagation();
          draggingNode = n;
          pointerMoved = false;
          group.setPointerCapture(event.pointerId);
          group.setAttribute("style", "cursor:grabbing");
        }});
        group.addEventListener("pointermove", (event) => {{
          if (!draggingNode || draggingNode.id !== n.id) return;
          const pt = clientToGraph(event.clientX, event.clientY);
          n.x = pt.x;
          n.y = pt.y;
          n.vx = 0;
          n.vy = 0;
          pointerMoved = true;
          render();
        }});
        group.addEventListener("pointerup", (event) => {{
          if (draggingNode && draggingNode.id === n.id) {{
            group.releasePointerCapture(event.pointerId);
            group.setAttribute("style", "cursor:grab");
            draggingNode = null;
            if (!pointerMoved) selectNode(n);
          }}
        }});
        group.addEventListener("dblclick", (event) => {{
          event.stopPropagation();
          centerOn(n);
        }});
        nodeLayer.appendChild(group);
        nodeElements.set(n.id, {{group, halo, circle, label}});
      }});

      function selectNode(n) {{
        selectedNodeId = n.id;
          const code = n.label_code;
          if (code && payload.details[code]) {{
            const d = payload.details[code];
            detail.innerHTML = `<strong>${{code}} - ${{d.title}}</strong><p><b>Symptoms:</b> ${{d.symptoms}}</p><p><b>Fundus:</b> ${{d.fundus}}</p><p><b>Tests:</b> ${{d.tests}}</p>`;
          }} else {{
            detail.innerHTML = `<strong>${{n.label}}</strong><p>图谱节点类型：${{n.kind}}。</p>`;
          }}
        render();
      }}

      function clientToGraph(clientX, clientY) {{
        const rect = svg.getBoundingClientRect();
        const x = (clientX - rect.left) * 1080 / rect.width;
        const y = (clientY - rect.top) * 660 / rect.height;
        return {{x:(x - tx) / scale, y:(y - ty) / scale}};
      }}
      function render() {{
        viewport.setAttribute("transform", `translate(${{tx}} ${{ty}}) scale(${{scale}})`);
        edgeElements.forEach(e => {{
          const a = byId[e.source], b = byId[e.target];
          const related = selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId);
          e.el.setAttribute("x1", a.x); e.el.setAttribute("y1", a.y);
          e.el.setAttribute("x2", b.x); e.el.setAttribute("y2", b.y);
          const searchRelated = searchQuery && (matchNode(a) || matchNode(b));
          e.el.setAttribute("stroke", related || searchRelated ? "#006b4e" : "#cddbd5");
          e.el.setAttribute("stroke-width", related || searchRelated ? "2" : "1");
          e.el.setAttribute("opacity", selectedNodeId ? (related ? "0.95" : "0.30") : (searchRelated ? "0.75" : "0.48"));
        }});
        nodes.forEach(n => {{
          const entry = nodeElements.get(n.id);
          const isNeighbor = selectedNodeId && neighborMap.get(selectedNodeId)?.has(n.id);
          const isSelectedNode = selectedNodeId === n.id;
          const isMatch = matchNode(n);
          const searchNeighbor = searchQuery && nodes.some(m => matchNode(m) && neighborMap.get(m.id)?.has(n.id));
          const isDimmed = selectedNodeId && !(isNeighbor || isSelectedNode);
          entry.group.setAttribute("transform", `translate(${{n.x}} ${{n.y}})`);
          entry.group.setAttribute("opacity", isDimmed ? "0.34" : (isMatch ? "1" : (searchNeighbor ? "0.78" : "0.54")));
          entry.circle.setAttribute("stroke-width", isSelectedNode || isMatch ? "3.2" : (isNeighbor || searchNeighbor ? "2.2" : "1.2"));
          entry.halo.setAttribute("opacity", isSelectedNode || isNeighbor || isMatch ? "0.82" : "0.38");
        }});
      }}
      function centerOn(n) {{
        scale = 1.35;
        tx = 540 - n.x * scale;
        ty = 330 - n.y * scale;
        render();
      }}
      let alpha = 1.0, simActive = true;
      function simStep() {{
        if (!simActive || alpha < 0.004) {{ simActive = false; return; }}
        alpha *= 0.993;
        nodes.forEach(n => {{ n.vx *= 0.80; n.vy *= 0.80; }});
        const repK = 3500 * alpha;
        for (let i = 0; i < nodes.length; i++) {{
          for (let j = i + 1; j < nodes.length; j++) {{
            const a = nodes[i], b = nodes[j];
            const dx = b.x - a.x, dy = b.y - a.y;
            const dist = Math.max(10, Math.hypot(dx, dy));
            const f = repK / (dist * dist);
            const fx = f * dx / dist, fy = f * dy / dist;
            a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
          }}
        }}
        const springLen = 130, springK = 0.016 * alpha;
        edgeElements.forEach(e => {{
          const a = byId[e.source], b = byId[e.target];
          const dx = b.x - a.x, dy = b.y - a.y;
          const dist = Math.max(1, Math.hypot(dx, dy));
          const f = (dist - springLen) * springK;
          const fx = f * dx / dist, fy = f * dy / dist;
          a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
        }});
        const gravK = 0.005 * alpha;
        nodes.forEach(n => {{ n.vx += (CX - n.x) * gravK; n.vy += (CY - n.y) * gravK; }});
        nodes.forEach(n => {{
          if (draggingNode && draggingNode.id === n.id) return;
          n.x = Math.max(35, Math.min(1045, n.x + n.vx));
          n.y = Math.max(35, Math.min(625, n.y + n.vy));
        }});
        render();
        requestAnimationFrame(simStep);
      }}
      function reheat(e) {{
        alpha = e || 0.72;
        if (!simActive) {{ simActive = true; requestAnimationFrame(simStep); }}
        else {{ alpha = Math.max(alpha, e || 0.72); }}
      }}
      svg.addEventListener("pointerdown", (event) => {{
        panning = {{x:event.clientX, y:event.clientY, tx, ty}};
        svg.setPointerCapture(event.pointerId);
      }});
      svg.addEventListener("pointermove", (event) => {{
        if (!panning || draggingNode) return;
        const rect = svg.getBoundingClientRect();
        tx = panning.tx + (event.clientX - panning.x) * 1080 / rect.width;
        ty = panning.ty + (event.clientY - panning.y) * 660 / rect.height;
        render();
      }});
      svg.addEventListener("pointerup", (event) => {{
        panning = null;
        svg.releasePointerCapture(event.pointerId);
      }});
      svg.addEventListener("wheel", (event) => {{
        event.preventDefault();
        const pt = clientToGraph(event.clientX, event.clientY);
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        const nextScale = Math.max(0.45, Math.min(2.8, scale * factor));
        const rect = svg.getBoundingClientRect();
        const sx = (event.clientX - rect.left) * 1080 / rect.width;
        const sy = (event.clientY - rect.top) * 660 / rect.height;
        scale = nextScale;
        tx = sx - pt.x * scale;
        ty = sy - pt.y * scale;
        render();
      }}, {{passive:false}});
      resetBtn.addEventListener("click", () => {{
        nodes.forEach((n, i) => {{
          const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2;
          n.x = CX + INIT_R * Math.cos(angle); n.y = CY + INIT_R * Math.sin(angle);
          n.vx = 0; n.vy = 0;
        }});
        scale = 1; tx = 0; ty = 0; selectedNodeId = null;
        detail.innerHTML = '<strong>点击选中节点</strong><p>点击疾病节点查看临床信息。</p>';
        reheat(1.0); render();
      }});
      settleBtn.addEventListener("click", () => reheat(0.8));
      if (searchQuery) {{
        const firstMatch = nodes.find(matchNode);
        if (firstMatch) {{
          selectNode(firstMatch);
          centerOn(firstMatch);
        }}
      }} else if (selected) {{
        const selectedNode = nodes.find(n => n.label_code === selected);
        if (selectedNode) selectNode(selectedNode);
      }}
      render();
      requestAnimationFrame(simStep);
    </script>
    """
    components.html(html, height=760, scrolling=False)


def render_review_summary(result_df: pd.DataFrame):
    sorted_df = result_df.sort_values("Probability", ascending=False).reset_index(drop=True)
    top = sorted_df.iloc[0]
    positives = result_df[result_df["Prediction"] == 1]
    if len(positives):
        action_label = positives.sort_values("Probability", ascending=False).iloc[0]["Label"]
        action = DISEASE_ACTIONS[action_label]
        positive_text = ", ".join([f"{row.Label} ({row.Probability:.2f})" for row in positives.itertuples()])
    else:
        action_label = "N"
        action = DISEASE_ACTIONS["N"]
        positive_text = "无阳性疾病标签"
    suppressed = result_df[(result_df["Prediction"] == 0) & (result_df["Probability"] >= 0.7)]
    suppressed_text = ""
    if len(suppressed):
        suppressed_text = "需复核的高分阴性项：" + ", ".join(
            [f"{row.Label} ({row.Probability:.2f})" for row in suppressed.itertuples()]
        )
    st.markdown(
        f"""
        <div class="review-summary">
            <div class="review-cell"><span>最高模型信号</span><strong>{top["Label"]} - {top["Disease"]} ({top["Probability"]:.2f})</strong></div>
            <div class="review-cell"><span>阳性标签</span><strong>{positive_text}</strong></div>
            <div class="review-cell"><span>建议复核重点</span><strong>{action}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if suppressed_text:
        st.caption(suppressed_text)


def render_chat_message(message, user_avatar_b64, ai_avatar_b64):
    role = "user" if message["role"] == "user" else "assistant"
    label = "患者" if role == "user" else "AI 眼科助手"
    with st.chat_message(role):
        st.caption(label)
        st.markdown(message["content"])


def load_feedback_review_records():
    records = []
    source_records = load_local_feedback()

    for data in source_records:
        if data.get("correct") and not data.get("flag_for_review", False):
            continue
        probs = data.get("probs", [0] * len(LABELS))
        preds = data.get("preds", [0] * len(LABELS))
        records.append(
            {
                "data": data,
                "pid": str(data.get("img_id") or data.get("patient_id") or ""),
                "max_prob": max(probs) if probs else 0,
                "positive_labels": [label for label, pred in zip(LABELS, preds) if int(pred) == 1],
                "flag_for_review": bool(data.get("flag_for_review", False)),
                "confidence": data.get("confidence"),
            }
        )
    return records, "file"


def errors_tab():
    records, feedback_source = load_feedback_review_records()
    source_label = "本地反馈文件" if feedback_source == "file" else feedback_source
    render_page_header(
        "错误病例与复查队列",
        "集中查看被标记为不正确或待复查的病例，便于后续复盘模型表现和人工标注质量。",
        [f"来源：{source_label}", f"记录 {len(records)} 条"],
    )
    if not records:
        st.markdown('<div class="empty-state">暂无被标记为错误或待复查的病例。</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    filter_cols = st.columns(4)
    with filter_cols[0]:
        label_filter = st.selectbox("疾病筛选", ["All"] + LABELS, format_func=lambda value: "全部标签" if value == "All" else f"{value} - {DISEASE_NAMES[value]}")
    with filter_cols[1]:
        flagged_only = st.checkbox("仅显示标记病例")
    with filter_cols[2]:
        min_probability = st.slider("最低概率阈值", 0.0, 1.0, 0.0, 0.05)
    with filter_cols[3]:
        sort_mode = st.selectbox("排序方式", ["按概率从高到低", "患者编号", "审查者信心"])
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = [
        record
        for record in records
        if (label_filter == "All" or label_filter in record["positive_labels"])
        and (not flagged_only or record["flag_for_review"])
        and record["max_prob"] >= min_probability
    ]
    if sort_mode == "按概率从高到低":
        filtered.sort(key=lambda record: record["max_prob"], reverse=True)
    elif sort_mode == "审查者信心":
        filtered.sort(key=lambda record: record["confidence"] or 0, reverse=True)
    else:
        filtered.sort(key=lambda record: record["pid"])

    st.caption(f"显示 {len(filtered)} / {len(records)} 条错误病例。")
    for record in filtered:
        data = record["data"]
        pid = record["pid"]
        st.markdown('<div class="error-case-card">', unsafe_allow_html=True)
        st.subheader(f"患者编号：{pid}")
        reviewer = data.get("doctor_name")
        created_at = data.get("created_at")
        st.markdown(
            f"""
            <div class="meta-strip">
              <div class="meta-pill"><b>审查者</b><span>{html.escape(str(reviewer or "未填写"))}</span></div>
              <div class="meta-pill"><b>标注时间</b><span>{html.escape(str(created_at or "未知"))}</span></div>
              <div class="meta-pill"><b>病例状态</b><span>{"待复查" if record["flag_for_review"] else "错误标注"}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_left, col_right = st.columns(2)
        with col_left:
            left_path = TRAIN_IMAGES_DIR / f"{pid}_left.jpg"
            if left_path.exists():
                show_image(Image.open(left_path), caption="左眼", width=220)
        with col_right:
            right_path = TRAIN_IMAGES_DIR / f"{pid}_right.jpg"
            if right_path.exists():
                show_image(Image.open(right_path), caption="右眼", width=220)
        feedback_probs = list(data.get("probs", []))[: len(LABELS)]
        feedback_preds = list(data.get("preds", []))[: len(LABELS)]
        feedback_probs += [0.0] * (len(LABELS) - len(feedback_probs))
        feedback_preds += [0] * (len(LABELS) - len(feedback_preds))
        detail_df = pd.DataFrame(
            {
                "Label": LABELS,
                "Disease": [DISEASE_NAMES[label] for label in LABELS],
                "Probability": [round(float(value), 2) for value in feedback_probs],
                "Prediction": feedback_preds,
            }
        )
        render_risk_panel(detail_df)
        st.markdown(
            f"""
            <div class="meta-strip">
              <div class="meta-pill"><b>审查者信心</b><span>{html.escape(str(data.get("confidence", "未填写")))}</span></div>
              <div class="meta-pill"><b>标记状态</b><span>{"已标记" if data.get("flag_for_review") else "否"}</span></div>
              <div class="meta-pill"><b>最高概率</b><span>{record['max_prob']:.2f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if data.get("comment"):
            st.markdown(f"**临床备注：** {data['comment']}")
        st.markdown("</div>", unsafe_allow_html=True)


def ai_doctor_tab(user_avatar_b64: str, ai_avatar_b64: str):
    st.markdown(
        """
        <style>
          .consult-status {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 16px;
          }
          .consult-status-card {
            border: 1px solid #dbe7e2;
            background: #ffffff;
            padding: 13px 14px;
            min-height: 74px;
            border-radius: 8px;
          }
          .consult-status-card b {
            display: block;
            color: #102a27;
            font-size: 13px;
            margin-bottom: 5px;
          }
          .consult-status-card span {
            color: #667874;
            font-size: 12px;
            line-height: 1.55;
          }
          .consult-section-title {
            margin: 18px 0 8px;
            color: #102a27;
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 0;
          }
          .consult-note {
            border-left: 3px solid #006b4e;
            background: #f4faf7;
            border-radius: 8px;
            color: #526663;
            padding: 10px 12px;
            font-size: 12px;
            line-height: 1.6;
            margin: 10px 0 14px;
          }
          @media (max-width: 920px) {
            .consult-status { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_page_header(
        "AI 问诊助手",
        "辅助整理分诊备注、检查建议和随访问答；回答会结合本地眼科知识库与当前诊断上下文。",
        ["检索依据", "临床辅助", "非最终诊断"],
    )

    active_predictions = st.session_state.get("active_predictions", {})
    active_probabilities = st.session_state.get("active_probabilities", {})
    active_labels = [label for label, value in active_predictions.items() if value == 1]
    if active_labels:
        case_status = "已联动阳性标签：" + "，".join(f"{label}-{DISEASE_NAMES[label]}" for label in active_labels[:4])
    elif active_probabilities:
        top_label = max(active_probabilities, key=active_probabilities.get)
        case_status = f"已联动最高风险：{top_label}-{DISEASE_NAMES[top_label]}（{active_probabilities[top_label]:.2f}）"
    else:
        case_status = "未选择诊断病例，可直接输入症状进行分诊辅助。"

    rag_status = "检索依据已启用，提交后会自动匹配相关疾病、检查和转诊标准。" if _RAG_AVAILABLE else "检索依据暂不可用，回答将基于输入信息生成。"
    st.markdown(
        f"""
        <div class="consult-status">
          <div class="consult-status-card"><b>当前病例</b><span>{case_status}</span></div>
          <div class="consult-status-card"><b>检索依据</b><span>{rag_status}</span></div>
          <div class="consult-status-card"><b>安全边界</b><span>仅提供分诊沟通参考；红旗症状会优先提示立即就医。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
    col_age, col_sex, col_history = st.columns(3)
    with col_age:
        patient_age = st.number_input("年龄", min_value=0, max_value=120, value=50, key="patient_age")
    with col_sex:
        patient_sex = st.radio("性别", ["男", "女"], horizontal=True, key="patient_sex")
    with col_history:
        patient_history = st.text_input("既往病史", placeholder="高血压、糖尿病、手术史", key="patient_history")

    symptoms = st.text_area(
        "症状描述",
        placeholder="视物模糊、眼痛、眼红、飞蚊症、视野缺损、干涩等",
        height=120,
        key="symptoms",
    )
    additional_info = st.text_area(
        "其他检查发现",
        placeholder="眼压、视力、眼底发现、OCT 发现等",
        height=80,
        key="additional_info",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="consult-note">本问诊助手会在后台参考本地眼科知识库；相关内容仅作为辅助参考，不替代医生面诊和最终诊断。</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="consult-section-title">对话记录</div>', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        render_chat_message(message, user_avatar_b64, ai_avatar_b64)
    if st.session_state.awaiting_response:
        render_chat_message({"role": "assistant", "content": "正在分析病例，请稍候..."}, user_avatar_b64, ai_avatar_b64)

    col_send, col_clear, col_export = st.columns([2, 1, 1])
    with col_send:
        send_button = st.button("发送", type="primary", disabled=not symptoms.strip() or st.session_state.awaiting_response)
    with col_clear:
        clear_button = st.button("清空对话")
    with col_export:
        export_disabled = len(st.session_state.chat_history) == 0
        export_format = st.selectbox("导出格式", ["TXT", "JSON"])
        export_button = st.button("导出对话", disabled=export_disabled)

    if clear_button:
        st.session_state.chat_history = []
        st.session_state.awaiting_response = False
        st.rerun()

    if export_button and st.session_state.chat_history:
        if export_format == "JSON":
            download_data = json.dumps(st.session_state.chat_history, indent=2, ensure_ascii=False)
            file_ext = "json"
            mime = "application/json"
        else:
            chat_text = "AI 眼科问诊记录\n" + "=" * 48 + "\n\n"
            for msg in st.session_state.chat_history:
                role = "患者" if msg["role"] == "user" else "AI 眼科助手"
                chat_text += f"{role}:\n{msg['content']}\n\n" + "-" * 30 + "\n\n"
            download_data = chat_text
            file_ext = "txt"
            mime = "text/plain"
        st.download_button(
            label=f"下载 {export_format} 记录",
            data=download_data,
            file_name=f"ophthalmology_consultation_{time.strftime('%Y%m%d_%H%M%S')}.{file_ext}",
            mime=mime,
        )

    if send_button and symptoms.strip() and not st.session_state.awaiting_response:
        user_message = f"年龄：{patient_age} 岁；性别：{patient_sex}"
        if patient_history:
            user_message += f"；既往史：{patient_history}"
        user_message += f"\n\n症状描述：{symptoms}"
        if additional_info:
            user_message += f"\n\n其他检查发现：{additional_info}"
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.awaiting_response = True
        st.rerun()

    if st.session_state.awaiting_response and st.session_state.chat_history:
        latest_user_msg = st.session_state.chat_history[-1]["content"]
        patient_meta = st.session_state.get("active_patient_meta", {})
        context_labels = [label for label, value in active_predictions.items() if value == 1]
        if not context_labels and active_probabilities:
            context_labels = [max(active_probabilities, key=active_probabilities.get)]
        graph_context = {label: graph_neighbors(label) for label in context_labels}
        multimodal_context = build_ai_consult_multimodal_context(
            active_predictions=active_predictions,
            active_probabilities=active_probabilities,
            patient_meta=patient_meta,
            disease_names=DISEASE_NAMES,
            graph_neighbors=graph_context,
            form_age=patient_age,
            form_sex=patient_sex,
            patient_history=patient_history,
            symptoms=symptoms,
            additional_info=additional_info,
        )

        rag_context = ""
        rag_evidence = {"results": [], "sources": [], "top_terms": [], "citation_ids": []}
        if _RAG_AVAILABLE:
            rag_query = build_case_rag_query(symptoms, additional_info, patient_history, patient_age, patient_sex)
            try:
                rag_context = _rag_module.build_context(rag_query, n_results=4)
                rag_evidence = _rag_module.explain_query(rag_query, n_results=4)
                st.session_state["last_rag_evidence"] = rag_evidence
            except Exception:
                rag_context = ""
                st.session_state["last_rag_evidence"] = rag_evidence

        messages = build_consult_messages(
            st.session_state.chat_history,
            case_context=multimodal_context,
            evidence_context=rag_context,
        )
        response_placeholder = st.empty()
        try:
            full_response = call_deepseek_api_stream(messages, DEEPSEEK_API_KEY, response_placeholder)
            st.session_state["last_ai_mode"] = "deepseek"
        except Exception:
            full_response = (
                "AI 服务暂时不可用，请检查 DEEPSEEK_API_KEY、网络连接和模型配置后重试。"
            )
            response_placeholder.error(full_response)
            st.session_state["last_ai_mode"] = "error"
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.awaiting_response = False
        st.rerun()

    last_evidence = st.session_state.get("last_rag_evidence")
    if last_evidence:
        with st.expander("检索依据"):
            st.caption(
                "命中文档："
                + (", ".join(last_evidence.get("sources", [])) or "无")
                + "；关键词："
                + ("、".join(last_evidence.get("top_terms", [])[:8]) or "无")
            )
            for index, result in enumerate(last_evidence.get("results", [])[:4], 1):
                preview = str(result["text"]).replace("\n", " ")[:220]
                st.markdown(
                    f"- **[R{index}]** `{result['source']}` · "
                    f"引用 `{result.get('citation_id', result.get('chunk_id', ''))}` · "
                    f"score={result['score']}：{preview}..."
                )

    with st.expander("使用提示"):
        st.markdown(
            """
            - 描述症状的持续时间、严重程度和诱因。
            - 如有客观检查结果（视力、眼压、OCT、眼底发现），请一并填写。
            - 红旗征象（突发失明、剧烈眼痛、外伤、大量飞蚊或闪光）请立即就医，不要等待 AI 回复。
            - AI 回复用于辅助分诊参考，不替代面对面的临床评估和最终诊断。
            """
        )

def diagnosis_tab(meta_df: pd.DataFrame, colors):
    st.sidebar.markdown("### 推理设置")
    cuda_available = torch.cuda.is_available()
    gpu_requested = st.sidebar.checkbox("使用 GPU", value=cuda_available, disabled=not cuda_available)
    if not cuda_available:
        st.sidebar.caption("当前 Python 环境中 CUDA 不可用，推理将在 CPU 上运行。")
    device = "cuda" if gpu_requested and cuda_available else "cpu"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 图像输入")
    upload_opt = st.sidebar.radio("来源", ["上传图像", "使用训练图像目录"])

    images = []
    if upload_opt == "上传图像":
        left_file = st.sidebar.file_uploader("上传左眼眼底图像", type=["jpg", "jpeg", "png"], key="upload_left_eye")
        right_file = st.sidebar.file_uploader("上传右眼眼底图像", type=["jpg", "jpeg", "png"], key="upload_right_eye")
        if left_file or right_file:
            if left_file and right_file:
                left_img = Image.open(left_file).convert("RGB")
                right_img = Image.open(right_file).convert("RGB")
                case_name = f"{left_file.name} + {right_file.name}"
                images.append(
                    {
                        "name": case_name,
                        "merged_img": merge_eye_images(left_img, right_img),
                        "left_img": left_img,
                        "right_img": right_img,
                        "source": "upload",
                    }
                )
            else:
                st.sidebar.warning("请同时上传左眼和右眼图像，才能进行双眼拼接和推理。")
    else:
        available_ids = sorted({int(p.name.split("_")[0]) for p in TRAIN_IMAGES_DIR.glob("*_left.jpg")})
        patient_ids = st.sidebar.multiselect("选择患者编号", available_ids)
        for pid in patient_ids:
            left_path = TRAIN_IMAGES_DIR / f"{pid}_left.jpg"
            right_path = TRAIN_IMAGES_DIR / f"{pid}_right.jpg"
            if left_path.exists() and right_path.exists():
                images.append(
                    {
                        "name": pid,
                        "merged_img": image_merge(left_path, right_path),
                        "left_img": Image.open(left_path).convert("RGB"),
                        "right_img": Image.open(right_path).convert("RGB"),
                        "source": "train",
                    }
                )
            else:
                st.sidebar.warning(f"患者 {pid} 缺少左眼或右眼图像。")

    render_page_header(
        "眼底诊断工作台",
        "按照病例信息、图像审查、AI 风险结果和反馈记录完成一次可追溯的眼底病例审阅。",
        ["图像核验", "风险筛查", "审查反馈"],
    )

    if not images:
        st.markdown('<div class="empty-state">请在左侧栏选择或上传图像以开始诊断。</div>', unsafe_allow_html=True)
        return

    for case in images:
        name = case["name"]
        merged_img = case["merged_img"]
        left_img = case["left_img"]
        right_img = case["right_img"]
        source_label = "训练目录" if case.get("source") == "train" else "上传图像"
        patient_key = str(name).split("_")[0]
        has_patient_id = patient_key.isdigit() and int(patient_key) in meta_df.index
        age, sex = None, None
        if has_patient_id:
            age = meta_df.loc[int(patient_key), "Patient Age"]
            sex = meta_df.loc[int(patient_key), "Patient Sex"]
            header_title = f"患者 {patient_key}"
            header_meta = f"年龄：{age} 岁 | 性别：{sex}"
        else:
            header_title = "上传图像"
            header_meta = str(name)

        st.markdown(
            f"""
            <div class="clinical-card" style="margin-top:4px;">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
                <div>
                  <div style="color:#006b4e;font-size:12px;font-weight:850;letter-spacing:.08em;text-transform:uppercase;">Case Review</div>
                  <h3 style="margin:6px 0 4px;color:#102a27;font-size:22px;">{header_title}</h3>
                  <div style="color:#667874;font-size:13px;">{header_meta}</div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                  <span class="disease-chip">Device: {device.upper()}</span>
                  <span class="disease-chip">8-label screening</span>
                  <span class="disease-chip">Swin Grad-CAM</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="meta-strip">
              <div class="meta-pill"><b>病例来源</b><span>{html.escape(source_label)}</span></div>
              <div class="meta-pill"><b>推理设备</b><span>{device.upper()}</span></div>
              <div class="meta-pill"><b>筛查标签</b><span>{len(LABELS)} 类 ODIR 标签</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left_path = TRAIN_IMAGES_DIR / f"{patient_key}_left.jpg"
        right_path = TRAIN_IMAGES_DIR / f"{patient_key}_right.jpg"
        cache_key = prediction_cache_key(name, merged_img, device)
        if cache_key not in st.session_state:
            with st.spinner("正在运行模型推理，请稍候..."):
                st.session_state[cache_key] = predict_scores(merged_img, MODELS_DIR, device=device)
        prediction_result = st.session_state[cache_key]
        probs, preds = prediction_result.probs, prediction_result.preds

        result_df = pd.DataFrame(
            {
                "Label": LABELS,
                "Disease": [DISEASE_NAMES[label] for label in LABELS],
                "Probability": np.round(probs, 2),
                "Prediction": preds,
            }
        )

        st.markdown('<div class="section-header"><h2>图像审查</h2><p>先做目视核验，再解读模型结果，避免只看概率数字。</p></div>', unsafe_allow_html=True)
        img_left_col, img_right_col, img_merged_col = st.columns([0.95, 0.95, 2.1], gap="medium")
        with img_left_col:
            render_diagnostic_image(left_img, caption="左眼", max_width=240, max_height=220)
        with img_right_col:
            render_diagnostic_image(right_img, caption="右眼", max_width=240, max_height=220)
        with img_merged_col:
            render_diagnostic_image(merged_img, caption="双眼拼接图像", max_width=560, max_height=260)

        image_quality = assess_image_quality(merged_img)

        st.session_state["active_patient_id"] = str(patient_key)
        st.session_state["active_predictions"] = {label: int(pred) for label, pred in zip(LABELS, preds)}
        st.session_state["active_probabilities"] = {label: float(prob) for label, prob in zip(LABELS, probs)}
        left_keywords = ""
        right_keywords = ""
        if has_patient_id:
            left_keywords = meta_df.loc[int(patient_key)].get("Left-Diagnostic Keywords", "")
            right_keywords = meta_df.loc[int(patient_key)].get("Right-Diagnostic Keywords", "")
        st.session_state["active_patient_meta"] = {
            "age": age,
            "sex": sex,
            "left_keywords": left_keywords,
            "right_keywords": right_keywords,
        }
        st.session_state["active_quality_summary"] = image_quality["overall"]

        st.markdown('<div class="section-header"><h2>AI 分析结果</h2><p>风险条用于快速扫描，热图用于解释模型关注区域。</p></div>', unsafe_allow_html=True)
        result_col, cam_col = st.columns([1.12, 0.88], gap="large")
        with result_col:
            render_review_summary(result_df)
            render_risk_panel(result_df)
            render_comorbidity_alert(result_df)
            st.markdown(
                '<div class="quiet-note">模型输出仅供审查参考。若图像质量、症状或临床病史与结果相悖，应优先遵循临床判断。</div>',
                unsafe_allow_html=True,
            )

        with cam_col:
            st.markdown("##### 模型注意力热图（Swin Grad-CAM）")
            default_idx = int(np.argmax(probs))
            selected_cam_label = st.selectbox(
                "选择关注的疾病标签",
                LABELS,
                index=default_idx,
                format_func=lambda lbl: f"{lbl} - {DISEASE_NAMES[lbl]}",
                key=f"cam_label_{name}",
            )
            cam_index = LABELS.index(selected_cam_label)
            cam_cache_key = f"{cache_key}_cam_{selected_cam_label}"
            if cam_cache_key not in st.session_state:
                with st.spinner("正在生成所选标签的注意力热图..."):
                    st.session_state[cam_cache_key] = generate_cam(
                        merged_img,
                        selected_cam_label,
                        MODELS_DIR,
                        device=device,
                    )
            cam_result = st.session_state[cam_cache_key]
            if cam_result.image is not None:
                render_cam_image(
                    cam_result.image,
                    caption=f"Swin Grad-CAM - {DISEASE_NAMES[selected_cam_label]}",
                    max_width=460,
                    max_height=300,
                )
                st.caption(f"该标签预测概率：{float(probs[cam_index]):.1%}")
            else:
                render_cam_image(merged_img, caption="原始双眼图像", max_width=460, max_height=300)
                st.caption("注意力热图暂不可用，已显示原始图像。")

            with st.expander("运行信息"):
                st.caption(
                    f"模型版本：{prediction_result.model_version} · "
                    f"设备：{prediction_result.device.upper()} · "
                    f"TTA：{'开启' if prediction_result.tta_enabled else '关闭'} · "
                    f"推理：{prediction_result.inference_ms:.0f} ms · "
                    f"热图：{cam_result.generation_ms:.0f} ms"
                )

            st.markdown("##### 医生审查反馈")
            with st.form(key=f"feedback_{name}", clear_on_submit=False):
                doctor_ok = st.radio("临床审查结果", ["正确", "不正确"], horizontal=True, key=f"radio_{name}")
                confidence = st.slider("审查者信心", min_value=1, max_value=5, value=4, key=f"confidence_{name}")
                flag_for_review = st.checkbox("标记此病例待复查", key=f"flag_{name}")
                feedback_comment = st.text_area(
                    "临床备注",
                    placeholder="不一致原因、图像质量问题、疑似标签、随访安排等。",
                    key=f"comment_{name}",
                )
                submitted = st.form_submit_button("提交反馈", type="primary")
                if submitted:
                    try:
                        saved_feedback = save_feedback(
                            name,
                            probs,
                            preds,
                            doctor_ok == "正确",
                            confidence=confidence,
                            comment=feedback_comment,
                            flag_for_review=flag_for_review,
                        )
                        st.session_state[f"feedback_saved_{name}"] = saved_feedback
                        st.session_state.pop(f"feedback_error_{name}", None)
                    except Exception:
                        st.session_state.pop(f"feedback_saved_{name}", None)
                        st.session_state[f"feedback_error_{name}"] = True
            saved_feedback = st.session_state.get(f"feedback_saved_{name}")
            if saved_feedback:
                st.success("反馈已保存至本地记录。")
            elif st.session_state.get(f"feedback_error_{name}"):
                st.error("反馈保存失败，请检查当前存储配置后重试。")

        st.divider()


def render_hero():
    st.markdown(
        """
        <div class="clinical-hero">
          <div class="hero-layout">
            <div>
              <div class="hero-kicker">RetinaScope Eye Center Suite</div>
              <h1>AI 智能眼底诊断工作站</h1>
              <p>面向眼底筛查、临床复核和模型改进闭环，把双眼图像、风险标签、Swin Grad-CAM 热图、知识图谱和医生反馈放进同一条清晰工作流。</p>
              <div class="hero-chips">
                <span class="disease-chip">双眼眼底图像</span>
                <span class="disease-chip">8 类风险标签</span>
                <span class="disease-chip">知识图谱联动</span>
                <span class="disease-chip">反馈可追溯</span>
              </div>
            </div>
            <div class="hero-status-grid">
              <div class="stat-card"><div class="stat-label">核心流程</div><div class="stat-value">诊断审阅</div><div class="metric-hint">选择或上传眼底图像，运行模型推理并查看风险结果。</div></div>
              <div class="stat-card"><div class="stat-label">解释方式</div><div class="stat-value">图谱联动</div><div class="metric-hint">风险条、热图和知识图谱互相补位。</div></div>
              <div class="stat-card"><div class="stat-label">反馈闭环</div><div class="stat-value">医生标注</div><div class="metric-hint">保存正确性、置信度、复查标记和临床备注。</div></div>
              <div class="stat-card"><div class="stat-label">辅助问诊</div><div class="stat-value">检索问答</div><div class="metric-hint">结合症状、病史和本地知识库生成分诊参考。</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(meta_df: pd.DataFrame):
    render_hero()
    review_img = local_image_data_uri("assets/overview/archive-review.png")
    evidence_img = local_image_data_uri("assets/overview/evidence-graph.png")
    case_img = local_image_data_uri("assets/overview/case-review.png")
    st.markdown(
        f"""
        <div class="overview-panel">
          <div>
            <div class="overview-copy">
              <div class="section-kicker">Diagnosis Flow</div>
              <h2>系统诊断流程</h2>
              <p>系统围绕一次眼底病例审查展开：先输入双眼图像，再运行模型筛查，随后查看热图和知识图谱，最后保存人工反馈。</p>
            </div>
            <div class="overview-flow" style="margin-top:10px;">
              <div class="flow-item"><strong>01 图像输入</strong><span>上传眼底图像，或从训练目录选择患者病例。</span></div>
              <div class="flow-item"><strong>02 AI 筛查</strong><span>输出 8 类眼底疾病风险概率和阳性提示。</span></div>
              <div class="flow-item"><strong>03 结果解释</strong><span>结合热图、知识图谱和检索依据辅助复核。</span></div>
              <div class="flow-item"><strong>04 反馈记录</strong><span>保存错误标注、复查状态和备注信息。</span></div>
            </div>
          </div>
          <div>
            <div class="overview-carousel">
              <div class="overview-slide">
                <img src="{review_img}" alt="眼底图像审查工作台">
                <div class="overview-slide-caption"><strong>病例信息确认</strong><span>先确认病例背景与双眼图像来源。</span></div>
              </div>
              <div class="overview-slide">
                <img src="{evidence_img}" alt="知识图谱证据联动界面">
                <div class="overview-slide-caption"><strong>结果解释与复核</strong><span>将模型提示转化为可沟通、可复核的临床线索。</span></div>
              </div>
              <div class="overview-slide">
                <img src="{case_img}" alt="眼科病例复核界面">
                <div class="overview-slide-caption"><strong>反馈闭环</strong><span>保存审查结论、复查状态和人工标注记录。</span></div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config("RetinaScope", layout="wide")
    colors = inject_theme(True)
    inject_night_refinement(colors)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <span class="sidebar-brand-mark">RS</span>
              <span class="sidebar-brand-text">
                <div class="sidebar-brand-title">RetinaScope</div>
                <div class="sidebar-brand-subtitle">AI 智能眼底诊断工作站</div>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="quiet-copyright">{COPYRIGHT_TEXT}</div>', unsafe_allow_html=True)

    meta_df = load_metadata()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    user_avatar_b64 = image_to_base64(create_avatar(colors["user"], "patient"))
    ai_avatar_b64 = image_to_base64(create_avatar(colors["assistant"], "doctor"))

    tab_names = ["概览", "眼底诊断", "知识图谱", "错误病例", "AI 问诊"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        overview_tab(meta_df)
    with tabs[1]:
        diagnosis_tab(meta_df, colors)
    with tabs[2]:
        knowledge_tab()
    with tabs[3]:
        errors_tab()
    with tabs[4]:
        ai_doctor_tab(user_avatar_b64, ai_avatar_b64)


if __name__ == "__main__":
    main()

