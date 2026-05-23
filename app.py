import base64
import html
import io
import json
import os
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageStat
from wordcloud import WordCloud

from utils.logger import save_feedback
from utils.model import LABELS, predict
from utils.paths import FEEDBACK_DIR, LABEL_CSV, MODELS_DIR, TRAIN_IMAGES_DIR
from utils.storage import save_diagnosis_report
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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com"))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
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


def image_merge(left_path: Path, right_path: Path, enhance: bool = True) -> Image.Image:
    """Merge left and right eye images into one side-by-side RGB image."""
    left_img = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

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
    st.image(display, caption=caption, width=display.width)


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
                 style="max-width: {max_width}px; max-height: {max_height}px;">
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


def call_deepseek_api_stream(prompt: str, api_key: str, message_placeholder):
    """Call DeepSeek official chat completion API and stream the response into Streamlit."""
    if not api_key:
        return "DEEPSEEK_API_KEY is not configured. Set the environment variable and restart the app."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    try:
        response = requests.post(deepseek_chat_completions_url(), headers=headers, json=payload, stream=True, timeout=60)
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
                message_placeholder.markdown(full_response + "...")
                time.sleep(0.02)
        message_placeholder.markdown(full_response)
        return full_response
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again later."
    except requests.exceptions.RequestException as exc:
        return f"Network error: {exc}"
    except Exception as exc:
        return f"Response processing error: {exc}"


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
        "bg": "#f8f5ef",
        "surface": "#fffdf9",
        "surface_raised": "#ffffff",
        "text": "#35504d",
        "muted": "#6f7d79",
        "primary": "#4f9b8f",
        "secondary": "#7aa874",
        "accent": "#eef6f2",
        "sidebar": "#f1eee6",
        "form": "#fffdf9",
        "border": "#ddd8cc",
        "user": "#eef6f2",
        "assistant": "#f6f1e8",
        "warning": "#b7791f",
        "danger": "#b65a4a",
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
            background: {colors["bg"]};
            color: {colors["text"]};
        }}
        .block-container {{
            padding-top: 1.2rem;
            max-width: 1560px;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}
        header[data-testid="stHeader"] {{
            background: {colors["surface"]};
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
            background: {colors["sidebar"]};
            border-right: 1px solid {colors["border"]};
        }}
        section[data-testid="stSidebar"] * {{
            color: {colors["text"]} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            margin-bottom: 10px;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 2px;
            border-bottom: 2px solid {colors["border"]};
            padding-bottom: 0;
        }}
        button[data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            background: transparent;
            border: 0;
            color: {colors["muted"]};
            padding: 8px 11px;
            font-weight: 500;
            font-size: 13px;
            letter-spacing: 0;
            transition: color 0.18s ease, background 0.18s ease;
            white-space: nowrap;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: {colors["surface"]};
            border-bottom: 3px solid {colors["primary"]};
            color: {colors["text"]};
            font-weight: 650;
        }}
        button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
            color: {colors["primary"]};
            background: rgba(79,155,143,0.07);
        }}
        input::placeholder, textarea::placeholder {{
            color: #6f7d79 !important;
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
        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] [role="button"],
        div[data-baseweb="base-input"],
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input {{
            background: {colors["surface"]} !important;
            color: {colors["text"]} !important;
            border-color: {colors["border"]} !important;
            box-shadow: none !important;
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
            background: #fbf8f1 !important;
            border: 1px dashed #cfc7b8 !important;
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
            background: #f1eee6;
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
            border-bottom: 1px solid #eee8dd;
            vertical-align: top;
            white-space: nowrap;
        }}
        .warm-table tr:nth-child(even) td {{
            background: #fbf8f1;
        }}
        .warm-table tr:last-child td {{
            border-bottom: 0;
        }}
        .clinical-hero {{
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 12px;
            color: {colors["text"]};
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            box-shadow: none;
        }}
        .clinical-hero h1 {{
            margin: 0;
            font-size: 22px;
            letter-spacing: 0;
        }}
        .clinical-hero p {{
            margin: 4px 0 0;
            max-width: 760px;
            color: {colors["muted"]};
            font-size: 14px;
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
            box-shadow: none;
        }}
        .stat-card {{
            text-align: center;
            border-left-width: 1px;
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
        .risk-row {{
            display: grid;
            grid-template-columns: 42px minmax(130px, 1.1fr) minmax(150px, 2fr) 58px;
            gap: 12px;
            align-items: center;
            padding: 8px 10px;
            margin-bottom: 6px;
            border: 1px solid {colors["border"]};
            border-radius: 6px;
            background: {colors["surface"]};
        }}
        .risk-row.positive {{
            border-color: #b7d8ce;
            background: #f0faf5;
        }}
        .risk-row.high {{
            border-color: #b7d8ce;
        }}
        .risk-row.medium {{
            border-color: #e2c889;
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
            background: #e8e3d8;
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
            border-left: 4px solid {colors["secondary"]};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 10px 0;
            background: #f0faf5;
            color: {colors["text"]};
        }}
        .stButton > button {{
            border-radius: 22px;
            border: 1.5px solid {colors["border"]};
            background: {colors["surface"]};
            color: {colors["text"]};
            min-height: 36px;
            padding: 0 20px;
            font-weight: 500;
            font-size: 13.5px;
            letter-spacing: 0.01em;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            border-color: {colors["primary"]};
            color: {colors["primary"]};
            background: rgba(79,155,143,0.06);
            box-shadow: 0 2px 10px rgba(79,155,143,0.15);
        }}
        button[kind="primary"] {{
            background: linear-gradient(135deg, {colors["primary"]} 0%, {colors["secondary"]} 100%) !important;
            border: 0 !important;
            border-radius: 22px !important;
            color: white !important;
            font-weight: 650 !important;
            box-shadow: 0 3px 14px rgba(79,155,143,0.35) !important;
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


def render_hero():
    carousel_html = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    background: transparent;
    overflow: hidden;
}
.carousel-wrap {
    position: relative;
    width: 100%;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 6px 40px rgba(20, 80, 90, 0.14);
    height: 380px;
    margin-bottom: 4px;
}
.carousel-track {
    display: flex;
    height: 100%;
    transition: transform 0.68s cubic-bezier(0.35, 0, 0.15, 1);
    will-change: transform;
}
.carousel-slide {
    min-width: 100%;
    height: 100%;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
/* Slide backgrounds */
.s1 { background: linear-gradient(135deg, #daeef7 0%, #b2ddf0 45%, #84c8e8 100%); }
.s2 { background: linear-gradient(135deg, #d5f0e4 0%, #a8e0c6 45%, #74caa4 100%); }
.s3 { background: linear-gradient(135deg, #d8eef0 0%, #a8d8dd 45%, #78bfc8 100%); }

/* Decorative SVG bg */
.slide-deco {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}
/* Right-side illustration */
.slide-illo {
    position: absolute;
    right: 60px;
    bottom: 0;
    opacity: 0.11;
    pointer-events: none;
}
/* Content */
.slide-content {
    position: relative;
    z-index: 3;
    text-align: center;
    padding: 0 52px;
    max-width: 740px;
}
.slide-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 7px 18px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 22px;
    color: #0e5a72;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.slide-title {
    font-size: 40px;
    font-weight: 800;
    line-height: 1.12;
    margin-bottom: 16px;
    letter-spacing: -0.025em;
}
.s1 .slide-title { color: #0a3d52; }
.s2 .slide-title { color: #0a4030; }
.s3 .slide-title { color: #0a3c42; }
.slide-desc {
    font-size: 15.5px;
    line-height: 1.72;
    max-width: 530px;
    margin: 0 auto;
}
.s1 .slide-desc { color: #1a6080; }
.s2 .slide-desc { color: #1a5a44; }
.s3 .slide-desc { color: #1a5060; }

/* Nav buttons */
.nav-btn {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: none;
    font-size: 26px;
    line-height: 1;
    cursor: pointer;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.22s ease;
    color: #0a4055;
    box-shadow: 0 2px 18px rgba(0,0,0,0.13);
    -webkit-user-select: none;
    user-select: none;
}
.nav-btn:hover {
    background: rgba(255,255,255,0.95);
    box-shadow: 0 4px 28px rgba(0,0,0,0.18);
    transform: translateY(-50%) scale(1.07);
}
.prev-btn { left: 22px; }
.next-btn { right: 22px; }

/* Indicators */
.indicators {
    position: absolute;
    bottom: 22px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 9px;
    z-index: 20;
}
.dot {
    height: 8px;
    width: 8px;
    border-radius: 4px;
    background: rgba(255,255,255,0.48);
    cursor: pointer;
    transition: all 0.32s ease;
}
.dot.active {
    background: rgba(255,255,255,0.93);
    width: 30px;
}

@media (max-width: 768px) {
    .carousel-wrap { height: 240px; border-radius: 12px; }
    .slide-title { font-size: 22px; }
    .slide-desc { font-size: 13px; }
    .slide-content { padding: 0 24px; }
    .slide-badge { font-size: 10px; padding: 5px 13px; margin-bottom: 14px; }
    .nav-btn { width: 38px; height: 38px; font-size: 20px; }
    .prev-btn { left: 10px; }
    .next-btn { right: 10px; }
    .slide-illo { display: none; }
}
</style>
</head>
<body>
<div class="carousel-wrap">
  <div class="carousel-track" id="track">

    <!-- Slide 1: Blue — AI Analysis -->
    <div class="carousel-slide s1">
      <svg class="slide-deco" viewBox="0 0 1200 380" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <circle cx="80" cy="55" r="160" fill="#7dd3fc" opacity="0.38"/>
        <circle cx="1150" cy="330" r="200" fill="#38bdf8" opacity="0.28"/>
        <circle cx="920" cy="40" r="100" fill="#0ea5e9" opacity="0.18"/>
        <circle cx="240" cy="350" r="70" fill="#7dd3fc" opacity="0.22"/>
        <path d="M0 195 Q300 175 600 195 Q900 215 1200 195" stroke="#38bdf8" stroke-width="1.2" fill="none" opacity="0.35"/>
        <path d="M0 215 Q300 235 600 215 Q900 195 1200 215" stroke="#38bdf8" stroke-width="0.8" fill="none" opacity="0.22"/>
        <circle cx="600" cy="30" r="4" fill="#0ea5e9" opacity="0.4"/>
        <circle cx="750" cy="350" r="5" fill="#38bdf8" opacity="0.35"/>
        <circle cx="400" cy="20" r="3" fill="#7dd3fc" opacity="0.5"/>
      </svg>
      <svg class="slide-illo" width="260" height="260" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="50" cy="50" rx="46" ry="30" stroke="#0a3d52" stroke-width="3.5" fill="none"/>
        <circle cx="50" cy="50" r="18" stroke="#0a3d52" stroke-width="3" fill="none"/>
        <circle cx="50" cy="50" r="9" fill="#0a3d52"/>
        <circle cx="43" cy="43" r="3.5" fill="white" opacity="0.6"/>
        <line x1="50" y1="5" x2="50" y2="20" stroke="#0a3d52" stroke-width="2" opacity="0.5"/>
        <line x1="50" y1="80" x2="50" y2="95" stroke="#0a3d52" stroke-width="2" opacity="0.5"/>
        <line x1="5" y1="50" x2="20" y2="50" stroke="#0a3d52" stroke-width="2" opacity="0.5"/>
        <line x1="80" y1="50" x2="95" y2="50" stroke="#0a3d52" stroke-width="2" opacity="0.5"/>
      </svg>
      <div class="slide-content">
        <div class="slide-badge">&#10008;&nbsp; AI 辅助诊断</div>
        <div class="slide-title">智能眼底影像分析系统</div>
        <div class="slide-desc">融合深度学习与临床医学知识，精准辅助医生识别眼部疾病，让每一张眼底照片都得到最严谨的审视与关怀</div>
      </div>
    </div>

    <!-- Slide 2: Green — Disease Coverage -->
    <div class="carousel-slide s2">
      <svg class="slide-deco" viewBox="0 0 1200 380" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <circle cx="90" cy="65" r="170" fill="#86efac" opacity="0.38"/>
        <circle cx="1140" cy="320" r="210" fill="#4ade80" opacity="0.28"/>
        <circle cx="940" cy="45" r="110" fill="#22c55e" opacity="0.16"/>
        <circle cx="220" cy="340" r="75" fill="#86efac" opacity="0.22"/>
        <path d="M0 190 Q300 172 600 190 Q900 208 1200 190" stroke="#22c55e" stroke-width="1.2" fill="none" opacity="0.35"/>
        <path d="M0 210 Q300 228 600 210 Q900 192 1200 210" stroke="#22c55e" stroke-width="0.8" fill="none" opacity="0.22"/>
        <circle cx="580" cy="28" r="4" fill="#4ade80" opacity="0.45"/>
        <circle cx="760" cy="352" r="5" fill="#22c55e" opacity="0.35"/>
      </svg>
      <svg class="slide-illo" width="240" height="240" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <rect x="40" y="12" width="20" height="76" rx="5" fill="#0a4030"/>
        <rect x="12" y="40" width="76" height="20" rx="5" fill="#0a4030"/>
        <circle cx="50" cy="50" r="42" stroke="#0a4030" stroke-width="2" fill="none" opacity="0.25"/>
      </svg>
      <div class="slide-content">
        <div class="slide-badge">&#9670;&nbsp; 多病种覆盖</div>
        <div class="slide-title">八类眼部疾病精准识别</div>
        <div class="slide-desc">涵盖糖尿病视网膜病变、青光眼、白内障、AMD等8类常见眼部疾病，基于多折叠 ConvNeXt 模型全方位守护您的视力健康</div>
      </div>
    </div>

    <!-- Slide 3: Teal — Warm Care -->
    <div class="carousel-slide s3">
      <svg class="slide-deco" viewBox="0 0 1200 380" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <circle cx="85" cy="60" r="165" fill="#5eead4" opacity="0.38"/>
        <circle cx="1145" cy="325" r="205" fill="#2dd4bf" opacity="0.28"/>
        <circle cx="930" cy="42" r="105" fill="#14b8a6" opacity="0.18"/>
        <circle cx="230" cy="345" r="72" fill="#5eead4" opacity="0.22"/>
        <path d="M0 192 Q300 173 600 192 Q900 211 1200 192" stroke="#14b8a6" stroke-width="1.2" fill="none" opacity="0.38"/>
        <path d="M0 212 Q300 231 600 212 Q900 193 1200 212" stroke="#14b8a6" stroke-width="0.8" fill="none" opacity="0.22"/>
        <circle cx="590" cy="30" r="4" fill="#2dd4bf" opacity="0.45"/>
        <circle cx="755" cy="350" r="5" fill="#14b8a6" opacity="0.38"/>
      </svg>
      <svg class="slide-illo" width="250" height="250" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 88 C8 62 4 24 26 17 C37 13 46 20 50 30 C54 20 63 13 74 17 C96 24 92 62 50 88Z" fill="#0a3c42"/>
        <circle cx="50" cy="42" r="10" fill="white" opacity="0.25"/>
        <line x1="50" y1="32" x2="50" y2="52" stroke="white" stroke-width="2.5" opacity="0.4"/>
        <line x1="40" y1="42" x2="60" y2="42" stroke="white" stroke-width="2.5" opacity="0.4"/>
      </svg>
      <div class="slide-content">
        <div class="slide-badge">&#9829;&nbsp; 温暖医疗</div>
        <div class="slide-title">以患者为中心的临床体验</div>
        <div class="slide-desc">简洁直观的操作界面配合科学的临床工作流，让医生专注于判断与关怀，为患者带来更安心、更温暖的就医体验</div>
      </div>
    </div>

  </div>

  <button class="nav-btn prev-btn" onclick="move(-1)">&#8249;</button>
  <button class="nav-btn next-btn" onclick="move(1)">&#8250;</button>

  <div class="indicators">
    <div class="dot active" onclick="go(0)"></div>
    <div class="dot" onclick="go(1)"></div>
    <div class="dot" onclick="go(2)"></div>
  </div>
</div>

<script>
var cur = 0, total = 3, timer;
var track = document.getElementById('track');
var dots = document.querySelectorAll('.dot');

function update() {
    track.style.transform = 'translateX(-' + (cur * 100) + '%)';
    dots.forEach(function(d, i) { d.classList.toggle('active', i === cur); });
}
function move(dir) { cur = (cur + dir + total) % total; update(); reset(); }
function go(i) { cur = i; update(); reset(); }
function reset() {
    clearInterval(timer);
    timer = setInterval(function() { move(1); }, 5000);
}
reset();
</script>
</body>
</html>"""
    components.html(carousel_html, height=410, scrolling=False)


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
            background: #fbf8f1;
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
            border-left: 3px solid #d5ad55;
            background: #fff8eb;
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
            background: #fbf8f1;
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
            box-shadow: 0 2px 14px rgba(30,90,80,0.07);
        }}
        .feature-card:hover, .knowledge-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 32px rgba(30,90,80,0.13);
        }}
        .pipeline-step {{
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 2px 10px rgba(30,90,80,0.06);
        }}
        .pipeline-step:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 22px rgba(30,90,80,0.11);
        }}
        .section-header h2 {{
            font-size: 28px !important;
            font-weight: 750 !important;
            letter-spacing: -0.022em !important;
        }}
        .section-header p {{
            font-size: 15px !important;
            line-height: 1.65 !important;
        }}
        .stat-card {{
            box-shadow: 0 2px 14px rgba(30,90,80,0.07);
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(30,90,80,0.12);
        }}
        .stat-value {{
            font-size: 30px !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }}
        .handoff-card {{
            border-left: 4px solid {colors["primary"]} !important;
            box-shadow: 0 2px 14px rgba(30,90,80,0.07) !important;
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
            box-shadow: 0 2px 10px rgba(30,90,80,0.08);
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


def overview_tab(meta_df: pd.DataFrame):
    df_all = meta_df.reset_index()
    render_hero()
    st.markdown(
        """
        <div class="section-header">
            <h2>临床审查工作台</h2>
            <p>本系统以临床医生的审查节奏为核心：了解患者背景、检视图像、将模型输出作为第二意见参考，最后记录人工审查结论。</p>
        </div>
        <div class="handoff-card">
            <h3>晨间交班场景</h3>
            <p>医生打开任务队列，选择患者，查看双眼眼底图像，扫描疾病风险条，核验注意力热图，并留下简短的临床备注。模型辅助工作流程，而非替代临床判断。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card"><h3>以图像为起点</h3><p>界面将左眼、右眼、合并及增强视图与风险面板紧邻排列，便于在信任数字之前进行目视核验。</p></div>
            <div class="feature-card"><h3>将风险作为提示</h3><p>预测结果以便于扫描的风险条形式呈现，并按行高亮，仅作为审查线索，非最终诊断结论。</p></div>
            <div class="feature-card"><h3>形成闭环</h3><p>反馈记录正误情况、审查者信心、备注及标记，使模型错误转化为可追踪的改进病例。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-header">
            <h2>诊断流程</h2>
            <p>每个步骤刻意保持简洁，目标是降低认知切换负担，而非使界面更加复杂。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_patient = st.session_state.get("active_patient_id")
    if active_patient:
        st.success(f"当前审查对象：患者 {active_patient}。知识图谱将高亮该病例的预测阳性标签。")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipeline-step"><strong>1. 配对图像</strong><span>加载左右眼眼底照片。</span></div>
            <div class="pipeline-step"><strong>2. 合并视图</strong><span>标准化尺寸，生成双眼诊断画布。</span></div>
            <div class="pipeline-step"><strong>3. 推断标签</strong><span>运行疾病特异性 ConvNeXt 二元分类器。</span></div>
            <div class="pipeline-step"><strong>4. 解释焦点</strong><span>生成 ScoreCAM 注意力热图以供目视核查。</span></div>
            <div class="pipeline-step"><strong>5. 审查病例</strong><span>保存医生对错误或不确定预测的反馈。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cases, col_labels, col_models = st.columns(3)
    with col_cases:
        st.markdown(f'<div class="stat-card"><div class="stat-label">数据集病例</div><div class="stat-value">{len(df_all)}</div><div class="metric-hint">从标签元数据加载的行数。</div></div>', unsafe_allow_html=True)
    with col_labels:
        st.markdown(f'<div class="stat-card"><div class="stat-label">疾病标签</div><div class="stat-value">{len(LABELS)}</div><div class="metric-hint">ODIR 筛查分类数量。</div></div>', unsafe_allow_html=True)
    with col_models:
        model_count = len(list(MODELS_DIR.glob("best_*_fold5.pth")))
        st.markdown(f'<div class="stat-card"><div class="stat-label">已加载检查点</div><div class="stat-value">{model_count}</div><div class="metric-hint">可用的 Fold-5 模型文件数。</div></div>', unsafe_allow_html=True)


def knowledge_tab():
    st.markdown(
        """
        <div class="section-header">
            <h2>眼科疾病知识图谱</h2>
            <p>作为临床参考：风险因素、症状、检查项及 ODIR 标签集中呈现，便于审查者分析某条诊断路径被激活的原因。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_predictions = st.session_state.get("active_predictions", {})
    active_labels = {label for label, value in active_predictions.items() if value == 1}
    selected_label = st.selectbox("疾病详情面板", LABELS, format_func=lambda label: f"{label} - {DISEASE_NAMES[label]}")
    render_obsidian_graph(active_labels=active_labels, selected_label=selected_label)

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
    if active_labels:
        active_text = ", ".join([f"{label} - {DISEASE_NAMES[label]}" for label in sorted(active_labels)])
        st.info(f"当前患者图谱叠加层：{active_text}")

    st.markdown("### 疾病速查")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="knowledge-card"><h3>糖尿病视网膜病变</h3><p>关注微动脉瘤、出血灶、硬性渗出、黄斑水肿及新生血管改变。</p></div>
            <div class="knowledge-card"><h3>青光眼</h3><p>典型线索包括视盘凹陷扩大、视网膜神经纤维层变薄、眼压升高及视野缺损。</p></div>
            <div class="knowledge-card"><h3>白内障</h3><p>晶状体混浊可降低图像清晰度，引起进行性视力模糊、眩光及对比度下降。</p></div>
            <div class="knowledge-card"><h3>AMD（年龄相关性黄斑变性）</h3><p>黄斑玻璃膜疣、色素改变、地图样萎缩或新生血管改变可影响中心视力。</p></div>
            <div class="knowledge-card"><h3>高血压视网膜病变</h3><p>小动脉变细、动静脉交叉压迹、出血、棉绒斑及视盘水肿可提示病变严重程度。</p></div>
            <div class="knowledge-card"><h3>病理性近视</h3><p>可见豹纹状眼底、后巩膜葡萄肿、漆裂纹、萎缩或近视性脉络膜新生血管。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 筛查标签")
    chips = "".join([f'<span class="disease-chip">{label} - {DISEASE_NAMES[label]}</span>' for label in LABELS])
    st.markdown(chips, unsafe_allow_html=True)


def risk_level(probability: float, positive: bool) -> tuple[str, str]:
    if positive:
        return "Positive", "high"
    if probability >= 0.7:
        return "Suppressed", "medium"
    if probability >= 0.3:
        return "Review", "medium"
    return "Low", "low"


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
            "name": "Brightness",
            "status": "fail" if brightness < 35 else "warn" if brightness < 55 else "ok",
            "value": f"{brightness:.0f}/255",
            "note": "Image is very dark" if brightness < 35 else "Slightly dark" if brightness < 55 else "Acceptable",
        },
        {
            "name": "Contrast",
            "status": "warn" if contrast < 28 else "ok",
            "value": f"{contrast:.0f}",
            "note": "Low contrast may hide lesions" if contrast < 28 else "Acceptable",
        },
        {
            "name": "Resolution",
            "status": "warn" if min(width, height) < 512 else "ok",
            "value": f"{width} x {height}",
            "note": "Below model target size" if min(width, height) < 512 else "Acceptable",
        },
        {
            "name": "Black border",
            "status": "warn" if dark_border_ratio > 0.65 else "ok",
            "value": f"{dark_border_ratio:.0%}",
            "note": "Large dark border detected" if dark_border_ratio > 0.65 else "Acceptable",
        },
    ]
    overall = "fail" if any(item["status"] == "fail" for item in checks) else "warn" if any(item["status"] == "warn" for item in checks) else "ok"
    return {"overall": overall, "checks": checks}


def render_quality_panel(quality):
    label = {"ok": "Acceptable", "warn": "Review image quality", "fail": "Poor image quality"}[quality["overall"]]
    st.markdown(f"#### Image Quality Check: {label}")
    items = []
    for item in quality["checks"]:
        css = "" if item["status"] == "ok" else item["status"]
        items.append(
            f'<div class="quality-item {css}"><strong>{item["name"]}</strong><span>{item["value"]} - {item["note"]}</span></div>'
        )
    st.markdown(f'<div class="quality-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def render_comorbidity_alert(result_df: pd.DataFrame):
    positives = result_df[result_df["Prediction"] == 1].sort_values("Probability", ascending=False)
    if len(positives) <= 1:
        return
    labels = [row.Label for row in positives.itertuples()]
    disease_text = ", ".join([f"{row.Label} - {row.Disease}" for row in positives.itertuples()])
    suggestions = []
    if {"D", "H"}.issubset(labels):
        suggestions.append("review systemic vascular risk, blood pressure, and diabetes control together")
    if {"D", "A"}.issubset(labels) or {"D", "M"}.issubset(labels):
        suggestions.append("consider macular OCT because central retinal complications may overlap")
    if {"G", "M"}.issubset(labels):
        suggestions.append("interpret optic disc findings carefully because high myopia can mimic glaucoma cues")
    if not suggestions:
        suggestions.append("prioritize clinician review because multiple disease labels are active")
    st.markdown(
        f"""
        <div class="insight-note">
            <strong>Multi-label review cue</strong><br>
            Active labels: {disease_text}. Suggested review: {"; ".join(suggestions)}.
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


def build_diagnosis_report(patient_key, age, sex, result_df: pd.DataFrame) -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sorted_df = result_df.sort_values("Probability", ascending=False).reset_index(drop=True)
    positives = result_df[result_df["Prediction"] == 1].sort_values("Probability", ascending=False)
    top = sorted_df.iloc[0]
    flagged = ", ".join([f"{row.Label} - {row.Disease} ({row.Probability:.2f})" for row in positives.itertuples()])
    if not flagged:
        flagged = "No positive disease label"
    action_label = positives.iloc[0]["Label"] if len(positives) else "N"
    lines = [
        "# RetinaScope Clinical Review Report",
        "",
        f"Generated: {timestamp}",
        f"Patient ID: {patient_key}",
        f"Age: {age if age is not None else 'Unknown'}",
        f"Sex: {sex if sex is not None else 'Unknown'}",
        "",
        "## Summary",
        f"- Highest raw model score: {top['Label']} - {top['Disease']} ({top['Probability']:.2f})",
        f"- Final positive label: {flagged}",
        f"- Suggested next check: {DISEASE_ACTIONS[action_label]}",
        "",
        "## Model Results",
        "| Label | Disease | Probability | Risk level | Prediction |",
        "|---|---|---:|---|---:|",
    ]
    for row in result_df.itertuples():
        level, _ = risk_level(float(row.Probability), int(row.Prediction) == 1)
        lines.append(f"| {row.Label} | {row.Disease} | {row.Probability:.2f} | {level} | {int(row.Prediction)} |")
    lines.extend(
        [
            "",
            "## Clinical Notes",
            "- This report is generated by an AI-assisted screening tool and should be reviewed by a qualified clinician.",
            "- Prioritize urgent care if symptoms include sudden vision loss, severe pain, trauma, flashes, floaters, or rapidly worsening vision.",
            "- Check image quality and clinical history before acting on a model prediction.",
        ]
    )
    return "\n".join(lines)


def render_report_panel(report_text: str, patient_key: str):
    st.markdown(
        """
        <div class="handoff-card">
            <h3>Generated clinical report</h3>
            <p>A structured Markdown report has been prepared from the model output. Review it before sharing or storing it in a clinical record.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download report as Markdown",
        data=report_text,
        file_name=f"retinascope_report_{patient_key}_{time.strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )
    with st.expander("Preview report"):
        st.markdown(report_text)


def render_obsidian_graph(active_labels=None, selected_label=None):
    active_labels = sorted(active_labels or [])
    payload = {
        "nodes": GRAPH_NODES,
        "edges": [{"source": source, "target": target} for source, target in GRAPH_EDGES],
        "activeLabels": active_labels,
        "selectedLabel": selected_label or "",
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
      #retina-graph-wrap {{ background: #f8f7f5; border-radius: 14px; padding: 18px; border: 1px solid #e7e2dc; }}
      .graph-shell {{ position: relative; background: #fffdfb; border: 1px solid #e6e0d9; border-radius: 12px; overflow: hidden; box-shadow: 0 14px 36px rgba(122, 94, 53, 0.10); }}
      .graph-topbar {{ height: 48px; display: flex; align-items: center; justify-content: space-between; padding: 0 16px; border-bottom: 1px solid #eee8e1; color: #35504d; font-family: Inter, Segoe UI, sans-serif; }}
      .graph-topbar strong {{ display: block; font-size: 14px; }}
      .graph-topbar span {{ color: #8a8178; font-size: 12px; }}
      .graph-actions {{ display: flex; gap: 8px; }}
      .graph-actions button {{ border: 1px solid #d8d1c8; background: #fffdf9; color: #5f736f; border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer; }}
      .graph-actions button:hover {{ background: #f3f0ec; }}
      .graph-legend {{ display: flex; gap: 12px; align-items: center; }}
      .graph-legend span {{ display: flex; gap: 5px; align-items: center; }}
      .graph-legend i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
      .graph-legend .core {{ background: #6ea99b; }}
      .graph-legend .disease {{ background: #8fbf89; }}
      .graph-legend .risk {{ background: #d5ad55; }}
      .graph-legend .test {{ background: #94aaa1; }}
      #retina-graph {{ width: 100%; height: 620px; display: block; background: radial-gradient(circle at 50% 45%, #ffffff, #fbfaf8); }}
      .graph-detail {{ position: absolute; right: 14px; bottom: 14px; width: 270px; max-height: 210px; overflow: auto; padding: 12px; border-radius: 10px; background: rgba(255,253,249,0.92); border: 1px solid #e6e0d9; color: #35504d; font-family: Inter, Segoe UI, sans-serif; box-shadow: 0 10px 26px rgba(122, 94, 53, 0.10); }}
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

      function matchNode(n) {{ return true; }}
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
        const line = make("line", {{"data-source":e.source, "data-target":e.target, stroke:"#d8d1c8", "stroke-width":"1", opacity:"0.55"}});
        edgeLayer.appendChild(line);
        edgeElements.push({{...e, el: line}});
      }});
      nodes.forEach(n => {{
        const isActive = n.label_code && active.has(n.label_code);
        const isSelected = n.label_code && n.label_code === selected;
        const isMatch = matchNode(n);
        const group = make("g", {{"data-id":n.id, opacity: isMatch ? "1" : "0.16", style:"cursor:grab"}});
        const halo = make("circle", {{r:n.r + (isActive || isSelected ? 9 : 4), fill:isActive ? "#eef6f2" : "#f1eee6", opacity:isActive || isSelected ? "0.82" : "0.34"}});
        const circle = make("circle", {{r:n.r, fill:isSelected ? "#4f9b8f" : isActive ? "#7aa874" : colors[n.kind] || "#94aaa1", stroke:"#fffdf9", "stroke-width": isActive || isSelected ? "2.4" : "1.2"}});
        const label = make("text", {{x:n.r + 5, y:4, fill:"#5f736f", "font-size": n.kind === "core" ? "12" : "9", "font-family":"Inter, Segoe UI, sans-serif", class:"node-label"}});
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
          e.el.setAttribute("stroke", related ? "#4f9b8f" : "#d8d1c8");
          e.el.setAttribute("stroke-width", related ? "2" : "1");
          e.el.setAttribute("opacity", selectedNodeId ? (related ? "0.95" : "0.18") : "0.55");
        }});
        nodes.forEach(n => {{
          const entry = nodeElements.get(n.id);
          const isNeighbor = selectedNodeId && neighborMap.get(selectedNodeId)?.has(n.id);
          const isSelectedNode = selectedNodeId === n.id;
          const isDimmed = selectedNodeId && !(isNeighbor || isSelectedNode);
          entry.group.setAttribute("transform", `translate(${{n.x}} ${{n.y}})`);
          entry.group.setAttribute("opacity", isDimmed ? "0.22" : (matchNode(n) ? "1" : "0.16"));
          entry.circle.setAttribute("stroke-width", isSelectedNode ? "3.2" : (isNeighbor ? "2.2" : entry.circle.getAttribute("stroke-width")));
          entry.halo.setAttribute("opacity", isSelectedNode || isNeighbor ? "0.72" : entry.halo.getAttribute("opacity"));
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
      if (selected) {{
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
        positive_text = "No positive disease label"
    suppressed = result_df[(result_df["Prediction"] == 0) & (result_df["Probability"] >= 0.7)]
    suppressed_text = ""
    if len(suppressed):
        suppressed_text = " Suppressed high scores: " + ", ".join(
            [f"{row.Label} ({row.Probability:.2f})" for row in suppressed.itertuples()]
        )
    st.markdown(
        f"""
        <div class="review-summary">
            <div class="review-cell"><span>Highest model signal</span><strong>{top["Label"]} - {top["Disease"]} ({top["Probability"]:.2f})</strong></div>
            <div class="review-cell"><span>Final positive label</span><strong>{positive_text}</strong></div>
            <div class="review-cell"><span>Suggested next check</span><strong>{action}</strong></div>
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


def diagnosis_tab(meta_df: pd.DataFrame, colors):
    st.sidebar.markdown("### 推理设置")
    cuda_available = torch.cuda.is_available()
    gpu_requested = st.sidebar.checkbox("使用 GPU", value=cuda_available, disabled=not cuda_available)
    if not cuda_available:
        st.sidebar.caption("此 Python 环境中 CUDA 不可用，推理将在 CPU 上运行。")
    device = "cuda" if gpu_requested and cuda_available else "cpu"
    st.sidebar.markdown("---")
    st.title("眼底诊断")
    st.sidebar.markdown("### 图像输入")
    upload_opt = st.sidebar.radio("来源", ["上传图像", "使用训练图像目录"])

    images = []
    if upload_opt == "上传图像":
        files = st.sidebar.file_uploader("选择一张或多张眼底图像", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        for file in files or []:
            images.append((file.name, Image.open(file).convert("RGB")))
    else:
        available_ids = sorted({int(p.name.split("_")[0]) for p in TRAIN_IMAGES_DIR.glob("*_left.jpg")})
        patient_ids = st.sidebar.multiselect("选择患者编号", available_ids)
        for pid in patient_ids:
            left_path = TRAIN_IMAGES_DIR / f"{pid}_left.jpg"
            right_path = TRAIN_IMAGES_DIR / f"{pid}_right.jpg"
            if left_path.exists() and right_path.exists():
                images.append((pid, image_merge(left_path, right_path)))
            else:
                st.sidebar.warning(f"患者 {pid} 缺少左眼或右眼图像。")

    if not images:
        st.info("请在侧栏选择或上传图像以开始诊断。")
        return

    for name, merged_img in images:
        patient_key = str(name).split("_")[0]
        has_patient_id = patient_key.isdigit() and int(patient_key) in meta_df.index
        age, sex = None, None
        if has_patient_id:
            age = meta_df.loc[int(patient_key), "Patient Age"]
            sex = meta_df.loc[int(patient_key), "Patient Sex"]
            st.markdown(
                f'<div class="section-header"><h2>患者 {patient_key}</h2>'
                f'<p>年龄：{age} 岁 &nbsp;|&nbsp; 性别：{sex}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="section-header"><h2>上传图像</h2><p>{name}</p></div>',
                unsafe_allow_html=True,
            )

        left_path = TRAIN_IMAGES_DIR / f"{patient_key}_left.jpg"
        right_path = TRAIN_IMAGES_DIR / f"{patient_key}_right.jpg"

        with st.spinner("正在运行模型推理，请稍候..."):
            probs, preds, cams, cam_errors = predict(merged_img, MODELS_DIR, device=device)

        result_df = pd.DataFrame(
            {
                "Label": LABELS,
                "Disease": [DISEASE_NAMES[label] for label in LABELS],
                "Probability": np.round(probs, 2),
                "Prediction": preds,
            }
        )

        # ── Section 1：图像审查 ──────────────────────────────────────────
        st.markdown("#### 图像审查")
        img_left_col, img_right_col, img_merged_col = st.columns([1, 1, 2.2], gap="medium")
        with img_left_col:
            if left_path.exists():
                render_diagnostic_image(Image.open(left_path), caption="左眼", max_width=260, max_height=260)
            else:
                st.caption("左眼图像不存在")
        with img_right_col:
            if right_path.exists():
                render_diagnostic_image(Image.open(right_path), caption="右眼", max_width=260, max_height=260)
            else:
                st.caption("右眼图像不存在")
        with img_merged_col:
            render_diagnostic_image(merged_img, caption="双眼拼接图像", max_width=560, max_height=280)
        render_quality_panel(assess_image_quality(merged_img))

        st.divider()

        # ── Section 2：AI 分析 ───────────────────────────────────────────
        st.markdown("#### AI 分析结果")
        result_col, cam_col = st.columns([1.12, 0.88], gap="large")

        with result_col:
            render_review_summary(result_df)
            render_risk_panel(result_df)
            render_comorbidity_alert(result_df)
            if has_patient_id:
                truth = meta_df.loc[int(patient_key), LABELS].astype(int)
                truth_labels = [f"{label} - {DISEASE_NAMES[label]}" for label in LABELS if int(truth[label]) == 1]
                with st.expander("数据集参考标签"):
                    st.caption("来自 CSV 元数据，仅供对比，不参与模型预测。")
                    st.write(", ".join(truth_labels) if truth_labels else "无标签记录。")
            with st.expander("详细概率数据表"):
                render_warm_table(result_df)
            st.markdown(
                '<div class="quiet-note">模型输出仅供审查参考。若图像质量、症状或临床病史与结果相悖，应优先遵循临床判断。</div>',
                unsafe_allow_html=True,
            )

        with cam_col:
            st.markdown("##### 模型注意力热图（GradCAM++）")
            if cams:
                default_idx = int(np.argmax(probs))
                selected_cam_label = st.selectbox(
                    "选择关注的疾病标签",
                    LABELS,
                    index=default_idx,
                    format_func=lambda lbl: f"{lbl} — {DISEASE_NAMES[lbl]}",
                    key=f"cam_label_{name}",
                )
                cam_index = LABELS.index(selected_cam_label)
                if cam_index < len(cams):
                    cam_img = cams[cam_index]
                    render_cam_image(cam_img, caption=f"GradCAM++ · {DISEASE_NAMES[selected_cam_label]}", max_width=460, max_height=320)
                    if selected_cam_label in cam_errors:
                        st.caption(f"注意力图生成失败，显示原始图像。错误：{cam_errors[selected_cam_label]}")
                    else:
                        prob_val = float(probs[cam_index])
                        st.caption(f"该标签预测概率：{prob_val:.1%}")
            else:
                st.info("未生成注意力图。")

        st.divider()

        # ── Section 3：临床文档 ───────────────────────────────────────────
        st.markdown("#### 临床文档")
        st.session_state["active_patient_id"] = str(patient_key)
        st.session_state["active_predictions"] = {label: int(pred) for label, pred in zip(LABELS, preds)}
        report_text = build_diagnosis_report(patient_key, age, sex, result_df)
        report_saved_key = f"report_saved_{patient_key}_{hash(report_text)}"
        if report_saved_key not in st.session_state:
            st.session_state[report_saved_key] = save_diagnosis_report(patient_key, age, sex, probs, preds, report_text)

        report_col, feedback_col = st.columns([1, 1], gap="large")
        with report_col:
            render_report_panel(report_text, patient_key)
            st.caption(f"报告 ID：{st.session_state[report_saved_key]}")

        with feedback_col:
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
                    save_feedback(name, probs, preds, doctor_ok == "正确", confidence=confidence, comment=feedback_comment, flag_for_review=flag_for_review)
                    st.session_state[f"feedback_saved_{name}"] = True
            if st.session_state.get(f"feedback_saved_{name}", False):
                st.success("反馈已保存。")

        st.divider()


def errors_tab():
    st.header("医生标记为错误的病例")
    error_files = sorted(FEEDBACK_DIR.glob("*.json"))
    if not error_files:
        st.info("暂无被标记为错误的病例。")
        return

    records = []
    for file_path in error_files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"Could not read {file_path.name}: {exc}")
            continue
        if data.get("correct"):
            continue
        probs = data.get("probs", [0] * len(LABELS))
        preds = data.get("preds", [0] * len(LABELS))
        records.append(
            {
                "data": data,
                "pid": str(data["img_id"]),
                "max_prob": max(probs) if probs else 0,
                "positive_labels": [label for label, pred in zip(LABELS, preds) if int(pred) == 1],
                "flag_for_review": bool(data.get("flag_for_review", False)),
                "confidence": data.get("confidence"),
            }
        )

    if not records:
        st.info("存在反馈数据，但无病例被标记为错误。")
        return

    filter_cols = st.columns(4)
    with filter_cols[0]:
        label_filter = st.selectbox("疾病筛选", ["All"] + LABELS, format_func=lambda value: "全部标签" if value == "All" else f"{value} - {DISEASE_NAMES[value]}")
    with filter_cols[1]:
        flagged_only = st.checkbox("仅显示标记病例")
    with filter_cols[2]:
        min_probability = st.slider("最低概率阈值", 0.0, 1.0, 0.0, 0.05)
    with filter_cols[3]:
        sort_mode = st.selectbox("排序方式", ["按概率从高到低", "患者编号", "审查者信心"])

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
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("审查者信心", data.get("confidence", "未填写"))
        with meta_cols[1]:
            st.metric("标记状态", "已标记" if data.get("flag_for_review") else "否")
        with meta_cols[2]:
            st.metric("最高概率", f"{record['max_prob']:.2f}")
        if data.get("comment"):
            st.markdown(f"**临床备注：** {data['comment']}")
        st.markdown("</div>", unsafe_allow_html=True)


def stats_tab(meta_df: pd.DataFrame, colors):
    st.markdown(
        """
        <div class="section-header">
            <h2>数据集概览</h2>
            <p>这些图表用于质量审查和偏差检验：年龄分布、性别比例、疾病流行率及多标签共现模式。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    df_all = meta_df.reset_index()

    col_total, col_age, col_male, col_female = st.columns(4)
    cards = [
        (col_total, "病例总数", len(df_all)),
        (col_age, "平均年龄", int(df_all["Patient Age"].mean())),
        (col_male, "男性病例", len(df_all[df_all["Patient Sex"] == "Male"])),
        (col_female, "女性病例", len(df_all[df_all["Patient Sex"] == "Female"])),
    ]
    for col, label, value in cards:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)

    top_left, top_right = st.columns(2)
    with top_left:
        st.subheader("年龄分布")
        ages = df_all["Patient Age"]
        bin_edges = list(range((ages.min() // 5) * 5, (ages.max() // 5 + 2) * 5, 5))
        age_counts = pd.cut(ages, bins=bin_edges, right=False).value_counts().sort_index()
        age_df = pd.DataFrame({"Age range": age_counts.index.map(str), "Cases": age_counts.values})
        age_chart = (
            alt.Chart(age_df)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Age range:N", title="Age range", axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("Cases:Q", title="Cases"),
                color=alt.value(colors["secondary"]),
                tooltip=["Age range", "Cases"],
            )
            .properties(width="container", height=300)
            .configure(background="transparent")
            .configure_axis(labelColor=colors["muted"], titleColor=colors["text"], gridColor=colors["border"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(age_chart, width="stretch")

    with top_right:
        st.subheader("性别比例")
        sex_df = df_all["Patient Sex"].value_counts().rename_axis("Sex").reset_index(name="Cases")
        sex_df["Percent"] = sex_df["Cases"] / sex_df["Cases"].sum()
        sex_chart = (
            alt.Chart(sex_df)
            .mark_arc(innerRadius=58, outerRadius=110)
            .encode(
                theta=alt.Theta("Cases:Q"),
                color=alt.Color("Sex:N", scale=alt.Scale(range=[colors["primary"], colors["secondary"]])),
                tooltip=["Sex", "Cases", alt.Tooltip("Percent:Q", format=".1%")],
            )
            .properties(width="container", height=300)
            .configure(background="transparent")
            .configure_legend(labelColor=colors["text"], titleColor=colors["text"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(sex_chart, width="stretch")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.subheader("各性别疾病标签分布")
        records = []
        for sex in ["Male", "Female"]:
            subset = df_all[df_all["Patient Sex"] == sex]
            for label in LABELS:
                records.append({"Label": label, "Sex": sex, "Count": int(subset[label].sum())})
        chart_df = pd.DataFrame(records)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Label:N", title="Disease label"),
                y=alt.Y("sum(Count):Q", title="Cases"),
                color=alt.Color("Sex:N", scale=alt.Scale(range=[colors["primary"], colors["secondary"]])),
                tooltip=["Label", "Sex", "Count"],
            )
            .properties(width="container", height=320)
            .configure(background="transparent")
            .configure_axis(labelColor=colors["muted"], titleColor=colors["text"], gridColor=colors["border"])
            .configure_legend(labelColor=colors["text"], titleColor=colors["text"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart.interactive(), width="stretch")

    with mid_right:
        st.subheader("疾病标签流行率")
        label_df = pd.DataFrame(
            {
                "Label": LABELS,
                "Disease": [DISEASE_NAMES[label] for label in LABELS],
                "Cases": [int(df_all[label].sum()) for label in LABELS],
            }
        )
        label_df["Prevalence"] = label_df["Cases"] / len(df_all)
        prevalence_chart = (
            alt.Chart(label_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Disease:N", sort="-x", title=None),
                x=alt.X("Cases:Q", title="Cases"),
                color=alt.Color("Prevalence:Q", scale=alt.Scale(range=[colors["primary"], colors["secondary"]])),
                tooltip=["Label", "Disease", "Cases", alt.Tooltip("Prevalence:Q", format=".1%")],
            )
            .properties(width="container", height=320)
            .configure(background="transparent")
            .configure_axis(labelColor=colors["muted"], titleColor=colors["text"], gridColor=colors["border"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(prevalence_chart, width="stretch")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("多标签共现热图")
        co_records = []
        for left in LABELS:
            for right in LABELS:
                co_records.append({"Left": left, "Right": right, "Cases": int(((df_all[left] == 1) & (df_all[right] == 1)).sum())})
        co_df = pd.DataFrame(co_records)
        co_chart = (
            alt.Chart(co_df)
            .mark_rect()
            .encode(
                x=alt.X("Left:N", title=None),
                y=alt.Y("Right:N", title=None),
                color=alt.Color("Cases:Q", scale=alt.Scale(range=[colors["accent"], colors["primary"], colors["secondary"]])),
                tooltip=["Left", "Right", "Cases"],
            )
            .properties(width="container", height=320)
            .configure(background="transparent")
            .configure_axis(labelColor=colors["text"], titleColor=colors["text"])
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(co_chart, width="stretch")

    with bottom_right:
        st.subheader("诊断关键词词云")
        word_cols = st.columns(2)
        for idx, (title, column) in enumerate([("左眼", "Left-Diagnostic Keywords"), ("右眼", "Right-Diagnostic Keywords")]):
            text = " ".join(df_all[column].dropna())
            if not text.strip():
                continue
            wc_img = WordCloud(width=340, height=230, background_color="#fffdf9", colormap="BuGn").generate(text)
            with word_cols[idx]:
                st.markdown(f"##### {title}")
                show_image(wc_img.to_array(), width=320)



def ai_doctor_tab(user_avatar_b64: str, ai_avatar_b64: str):
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <h2>AI 问诊助手</h2>
            <p>用于辅助整理分诊备注、检查建议和随访问答，不替代面对面的临床评估。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="feature-grid">
                <div class="feature-card"><h3>示例：急性症状</h3><p>突然出现飞蚊症和闪光感一天，应检查哪些危险信号？</p></div>
                <div class="feature-card"><h3>示例：慢性模糊</h3><p>进行性视物模糊，夜间眩光增强，对比度下降。</p></div>
                <div class="feature-card"><h3>示例：视网膜风险</h3><p>糖尿病史 12 年，新发中心视物模糊，OCT 提示黄斑水肿。</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    st.caption("本问诊助手已接入本地眼科知识库 RAG 检索；相关内容仅作为辅助参考，不替代医生面诊和最终诊断。")

    st.markdown("#### 对话记录")
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

        rag_context = ""
        if _RAG_AVAILABLE:
            rag_query = f"{symptoms} {additional_info} {patient_history}".strip()
            try:
                rag_context = _rag_module.build_context(rag_query, n_results=4)
            except Exception:
                rag_context = ""

        rag_section = (
            f"\n\n请参考以下本地知识库检索结果，但不要把它当作唯一依据；如果与患者信息不匹配，应说明不确定性。\n{rag_context}\n"
            if rag_context
            else "\n\n本次未检索到可用的本地知识库片段，请仅基于患者提供的信息给出谨慎建议。\n"
        )
        prompt = f"""你是一名眼科 AI 问诊辅助助手，只能提供分诊和沟通参考，不能作出最终诊断、不能替代医生面诊，也不能要求患者自行用药或延误就医。{rag_section}
患者信息：
{latest_user_msg}

请使用中文，语气专业、克制、易懂。严格按照以下格式输出，不要省略任何一级标题：

## 疾病风险等级
**辅助分诊风险**：高 / 中 / 低（三选一，加粗显示）
**判断依据**：（1-2 句话说明主要依据，并指出哪些信息仍不足）

## 建议检查项目
- （列出 3-5 项检查，并说明目的；优先包含视力、裂隙灯、眼压、眼底检查、OCT 等相关项目）

## 初步处理建议
（2-4 句话，说明就诊优先级、是否需要尽快眼科就诊、观察/复诊重点；不要给出处方）

## 需要立即就医的情况
- （列出与本病例相关的红旗症状；如突发视力下降、剧烈眼痛、外伤、闪光/大量飞蚊、视野缺损等）

## 给医生的沟通要点
- （帮助患者整理面诊时应提供的信息，如症状起始时间、单眼/双眼、基础病、既往手术、检查结果等）
---
本回答仅为 AI 辅助建议，不能替代专业医生面诊、检查和最终医疗判断。"""

        response_placeholder = st.empty()
        try:
            full_response = call_deepseek_api_stream(prompt, DEEPSEEK_API_KEY, response_placeholder)
        except Exception as exc:
            full_response = f"AI 服务暂时不可用：{exc}"
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.awaiting_response = False
        st.rerun()

    with st.expander("使用提示"):
        st.markdown(
            """
            - 描述症状的持续时间、严重程度和诱因。
            - 如有客观检查结果（视力、眼压、OCT、眼底发现），请一并填写。
            - 红旗征象（突发失明、剧烈眼痛、外伤、大量飞蚊或闪光）请立即就医，不要等待 AI 回复。
            - AI 回复用于辅助分诊参考，不替代面对面的临床评估和最终诊断。
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config("Ophthalmic AI Assistant", layout="wide")
    colors = inject_theme(True)
    inject_night_refinement(colors)

    meta_df = load_metadata()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    user_avatar_b64 = image_to_base64(create_avatar(colors["user"], "patient"))
    ai_avatar_b64 = image_to_base64(create_avatar(colors["assistant"], "doctor"))

    tab_overview, tab_main, tab_knowledge, tab_errors, tab_stats, tab_ai_doctor = st.tabs(
        ["概览", "眼底诊断", "知识图谱", "错误病例", "数据集统计", "AI 问诊"]
    )
    with tab_overview:
        overview_tab(meta_df)
    with tab_main:
        diagnosis_tab(meta_df, colors)
    with tab_knowledge:
        knowledge_tab()
    with tab_errors:
        errors_tab()
    with tab_stats:
        stats_tab(meta_df, colors)
    with tab_ai_doctor:
        ai_doctor_tab(user_avatar_b64, ai_avatar_b64)


if __name__ == "__main__":
    main()
