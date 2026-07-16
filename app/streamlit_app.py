"""
================================================================================
  THE HERBARIUM — A Botanical Diagnostic Atlas
  Plant Disease Classification, presented as a field-notebook / specimen index
================================================================================

Reuses: predict.py, gradcam.py, config.py, evaluate.py outputs from Phases 1-9.
Run:
    cd "Project CNN"
    streamlit run app/streamlit_app.py
================================================================================
"""

import sys
import csv
import json
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
from PIL import Image

from config import (
    MODELS_DIR, PLOTS_DIR, GRADCAM_DIR, MISCLASSIFIED_DIR,
    METADATA_DIR, RAW_DATA_DIR, SAMPLES_DIR, CLASS_MAPPING_JSON,
)
from predict import load_model, predict_image
from gradcam import GradCAM, denormalize_image, overlay_heatmap
from augmentations import get_eval_transforms


st.set_page_config(page_title="The Herbarium — Plant Diagnostics", page_icon="🌿", layout="wide")

# ============================================================================
# HERBARIUM THEME — CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --paper:      #EDE6D3;
        --paper-card: #F5F0E3;
        --paper-deep: #E3DAC3;
        --line:       #C9BFA0;
        --rule:       #B7AC8A;
        --ink:        #1B3328;
        --ink-soft:   #2E4A3B;
        --ink-faint:  #5A6E5F;
        --moss:       #4F7A5E;
        --moss-dim:   #3B5F47;
        --sage:       #8FAE86;
        --rust:       #A3492F;
        --gold:       #B4863B;
        --indigo:     #445C82;
        --radius-sm: 4px;
        --radius-md: 8px;
        --shadow-card: 0 2px 10px rgba(27,51,40,0.10), 0 1px 2px rgba(27,51,40,0.08);
    }

    html, body, [class*="css"], .stApp, .main, .block-container {
        background-color: var(--paper) !important;
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse 60% 40% at 90% 5%, rgba(79,122,94,0.06) 0%, transparent 55%),
            radial-gradient(ellipse 50% 35% at 5% 90%, rgba(163,73,47,0.04) 0%, transparent 55%),
            var(--paper) !important;
    }
    .block-container { padding: 2.2rem 3rem 4rem !important; max-width: 1500px; margin: 0 auto; }
    [data-testid="stSidebar"] {
        background: var(--paper-deep);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: var(--ink-soft) !important; font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] hr { border-color: var(--line); }

    h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }

    /* ── HERO ─────────────────────────────────────────────── */
    .field-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 500;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--moss);
        border: 1px solid var(--moss); display: inline-block; padding: 3px 10px;
        border-radius: 20px; margin-bottom: 0.7rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif; font-weight: 600; font-style: italic;
        font-size: clamp(2rem, 3.6vw, 3.1rem); line-height: 1.08; color: var(--ink);
        margin: 0 0 0.5rem 0;
    }
    .hero-sub { font-size: 1rem; color: var(--ink-faint); max-width: 620px; line-height: 1.65; }

    /* ── SPECIMEN CARD (signature element) ───────────────── */
    .specimen-card {
        background: var(--paper-card);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 1.3rem 1.4rem 1.1rem;
        position: relative;
        box-shadow: var(--shadow-card);
        height: 100%;
        box-sizing: border-box;
    }
    .specimen-card::before {
        content: ''; position: absolute; inset: 8px;
        border: 1px dashed var(--rule); border-radius: 3px; pointer-events: none;
    }
    .specimen-field-no {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--ink-faint);
        letter-spacing: 0.06em; margin-bottom: 0.5rem;
    }
    .specimen-title {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem;
        color: var(--ink); margin-bottom: 0.15rem;
    }
    .specimen-latin {
        font-family: 'Fraunces', serif; font-style: italic; font-size: 0.82rem;
        color: var(--ink-faint); margin-bottom: 0.7rem;
    }
    .specimen-stat-row { display: flex; justify-content: space-between; align-items: baseline; margin-top: 0.3rem; }
    .specimen-stat-label { font-size: 0.72rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.06em; }
    .specimen-stat-val { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.5rem; color: var(--moss-dim); }

    /* ── SECTION HEADER ──────────────────────────────────── */
    .section-label { display: flex; align-items: baseline; gap: 10px; margin: 2rem 0 1rem 0; }
    .section-label-mark { font-family: 'Fraunces', serif; font-style: italic; color: var(--moss); font-size: 1.1rem; }
    .section-label-text { font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 600; color: var(--ink); }
    .section-label-line { flex: 1; height: 1px; background: var(--rule); }

    /* ── FIELD NOTE / INSIGHT PANEL ──────────────────────── */
    .field-note {
        background: var(--paper-card); border: 1px solid var(--line);
        border-left: 3px solid var(--moss); border-radius: var(--radius-sm);
        padding: 0.9rem 1.2rem; margin: 0.5rem 0;
    }
    .field-note.warn   { border-left-color: var(--gold); }
    .field-note.danger { border-left-color: var(--rust); }
    .field-note.info   { border-left-color: var(--indigo); }
    .field-note-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--moss); margin-bottom: 0.3rem;
    }
    .field-note.warn .field-note-label   { color: var(--gold); }
    .field-note.danger .field-note-label { color: var(--rust); }
    .field-note.info .field-note-label   { color: var(--indigo); }
    .field-note-body { font-size: 0.88rem; color: var(--ink-soft); line-height: 1.6; }
    .field-note-body strong { color: var(--ink); }

    /* ── CAVEAT BANNER ────────────────────────────────────── */
    .caveat-banner {
        background: repeating-linear-gradient(135deg, var(--paper-deep), var(--paper-deep) 10px, var(--paper-card) 10px, var(--paper-card) 20px);
        border: 1px solid var(--gold); border-radius: var(--radius-md);
        padding: 0.9rem 1.3rem; margin: 1rem 0 1.5rem;
    }
    .caveat-banner strong { color: var(--rust); font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; display: block; margin-bottom: 3px; }
    .caveat-banner span { font-size: 0.87rem; color: var(--ink-soft); line-height: 1.55; }

    /* ── TABS ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: var(--paper-deep); border-radius: var(--radius-md); padding: 3px; border: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.86rem; padding: 7px 16px; }
    .stTabs [aria-selected="true"] { background: var(--moss) !important; border-radius: 6px; font-weight: 600; }

    /* ── DATAFRAMES / EXPANDERS ──────────────────────────── */
    .stDataFrame, [data-testid="stDataFrame"] { border: 1px solid var(--line) !important; border-radius: var(--radius-md) !important; background: var(--paper-card) !important; }
    [data-testid="stExpander"] summary { background: var(--paper-card) !important; border: 1px solid var(--line) !important; border-radius: var(--radius-sm) !important; font-weight: 600 !important; color: var(--ink) !important; }

    /* ── PROGRESS BAR ─────────────────────────────────────── */
    .stProgress > div > div { background-color: var(--moss) !important; }

    /* ── SIDEBAR LOGO ─────────────────────────────────────── */
    .nav-logo { padding: 1.4rem 1.2rem 1rem; border-bottom: 1px solid var(--line); }
    .nav-logo-mark { font-family: 'Fraunces', serif; font-style: italic; font-weight: 600; font-size: 1.4rem; color: var(--ink); }
    .nav-logo-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--ink-faint); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 3px; }
    .nav-section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-faint); padding: 1rem 1.2rem 0.3rem; }
    .sq-stat { margin: 0 0.7rem 6px; padding: 0.6rem 0.9rem; background: var(--paper-card); border: 1px solid var(--line); border-radius: var(--radius-sm); display: flex; justify-content: space-between; }
    .sq-label { font-size: 0.7rem; color: var(--ink-faint); }
    .sq-value { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.88rem; color: var(--moss-dim); }

    .ruled-divider { height: 1px; background: var(--rule); margin: 2rem 0; opacity: 0.7; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: var(--paper) !important; border-bottom: 1px solid var(--line) !important; }

    /* ── SIDEBAR NAV (radio) ──────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        margin: 0 0.6rem 2px;
        transition: background 0.15s ease;
        display: flex !important;
        align-items: center;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(79,122,94,0.10);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label span[data-testid="stMarkdownContainer"] p {
        color: var(--ink-soft) !important;
        font-weight: 500;
        font-size: 0.92rem;
    }
    /* the little selector dot itself */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        border-color: var(--moss) !important;
        background-color: var(--paper-card) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child > div {
        background-color: var(--moss) !important;
    }
    /* selected row gets a soft moss highlight + darker ink label */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[aria-checked="true"] {
        background-color: var(--sage) !important;
        border-color: var(--moss-dim) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: var(--sage);
        box-shadow: inset 0 0 0 1px var(--moss);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span[data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
        font-weight: 700;
    }

    /* ── TABS — unselected labels were inheriting white text ── */
    .stTabs [aria-selected="false"],
    .stTabs [aria-selected="false"] * {
        color: var(--ink-faint) !important;
    }
    .stTabs [aria-selected="false"]:hover,
    .stTabs [aria-selected="false"]:hover * {
        color: var(--moss-dim) !important;
    }
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] * {
        color: var(--paper) !important;
    }

    /* ── FILE UPLOADER — was using Streamlit's dark default theme ── */
    [data-testid="stFileUploader"] label p,
    [data-testid="stFileUploader"] label span {
        color: var(--ink) !important;
        font-weight: 500;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--paper-card) !important;
        border: 1px dashed var(--rule) !important;
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--ink-soft) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: var(--moss) !important;
    }
    [data-testid="stFileUploader"] button {
        background: var(--paper) !important;
        border: 1px solid var(--moss) !important;
        color: var(--moss-dim) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--moss) !important;
        color: var(--paper) !important;
    }
    [data-testid="stFileUploader"] button p { color: inherit !important; }
    [data-testid="stFileUploaderFile"] {
        background: var(--paper-card) !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-sm) !important;
    }
    [data-testid="stFileUploaderFile"] div,
    [data-testid="stFileUploaderFile"] span {
        color: var(--ink-soft) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PLOTLY — HERBARIUM THEME
# ============================================================================

PLOTLY_HERBARIUM = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#2E4A3B"),
    title_font=dict(family="Fraunces, serif", size=15, color="#1B3328"),
    margin=dict(l=48, r=24, t=48, b=42),
    colorway=["#4F7A5E", "#B4863B", "#A3492F", "#445C82", "#8FAE86", "#8C6D4F"],
    legend=dict(font=dict(color="#2E4A3B")),
    xaxis=dict(gridcolor="rgba(183,172,138,0.35)", linecolor="#C9BFA0", tickfont=dict(size=11, color="#2E4A3B"),
               title_font=dict(color="#1B3328")),
    yaxis=dict(gridcolor="rgba(183,172,138,0.35)", linecolor="#C9BFA0", tickfont=dict(size=11, color="#2E4A3B"),
               title_font=dict(color="#1B3328")),
)

def herb_chart(fig):
    fig.update_layout(**PLOTLY_HERBARIUM)
    return fig


# ============================================================================
# REUSABLE COMPONENTS
# ============================================================================

def section(title, mark="🌿"):
    st.markdown(f"""
    <div class="section-label">
        <span class="section-label-mark">{mark}</span>
        <span class="section-label-text">{title}</span>
        <div class="section-label-line"></div>
    </div>
    """, unsafe_allow_html=True)

def field_note(label, body, kind=""):
    st.markdown(f"""
    <div class="field-note {kind}">
        <div class="field-note-label">{label}</div>
        <div class="field-note-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

def caveat_banner():
    st.markdown("""
    <div class="caveat-banner">
        <strong>⚠ Field Limitation</strong>
        <span>Specimens in this atlas were photographed individually against plain,
        controlled backgrounds (the PlantVillage collection). Identification confidence
        may drop on field photographs with cluttered backgrounds, multiple leaves,
        or inconsistent lighting — treat those results as a starting point, not a diagnosis.</span>
    </div>
    """, unsafe_allow_html=True)

def divider():
    st.markdown('<div class="ruled-divider"></div>', unsafe_allow_html=True)

def specimen_stat_card(col, field_no, title, latin, value, suffix=""):
    col.markdown(f"""
    <div class="specimen-card">
        <div class="specimen-field-no">{field_no}</div>
        <div class="specimen-title">{title}</div>
        <div class="specimen-latin">{latin}</div>
        <div class="specimen-stat-row">
            <span class="specimen-stat-label">Value</span>
            <span class="specimen-stat-val">{value}{suffix}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def parse_class_name(class_name: str):
    """'Crop___Disease' -> ('Crop', 'Disease' or 'Healthy')"""
    crop, _, disease = class_name.partition("___")
    disease_display = disease.replace("_", " ") if disease else "Healthy"
    return crop, disease_display


# ============================================================================
# DATA LOADING (all optional — app degrades gracefully if a phase hasn't run)
# ============================================================================

@st.cache_data
def load_class_mapping():
    if CLASS_MAPPING_JSON.exists():
        with open(CLASS_MAPPING_JSON) as f:
            return json.load(f)
    return None

@st.cache_data
def load_class_distribution():
    path = METADATA_DIR / "class_distribution.csv"
    if path.exists():
        with open(path) as f:
            return list(csv.DictReader(f))
    return None

@st.cache_data
def load_evaluation_summary():
    path = METADATA_DIR / "evaluation_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_test_predictions():
    path = METADATA_DIR / "test_predictions.csv"
    if path.exists():
        with open(path) as f:
            return list(csv.DictReader(f))
    return None

@st.cache_data
def load_classification_report_text():
    path = PLOTS_DIR / "classification_report.txt"
    if path.exists():
        return path.read_text()
    return None

@st.cache_data
def load_history(model_name):
    from config import LOGS_DIR
    path = LOGS_DIR / f"{model_name}_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_resource
def get_model_and_gradcam():
    model, idx_to_class, device = load_model()
    target_layer = model.features[-1].block[0]
    gradcam = GradCAM(model, target_layer)
    return model, idx_to_class, device, gradcam


# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home(class_mapping, class_dist, eval_summary):
    st.markdown('<div class="field-label">🌿 Herbarium Atlas · Field Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">A specimen index for<br>plant leaf pathology</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-sub">Trained on the PlantVillage collection — 38 catalogued conditions across
    14 crop species. Upload a leaf under <em>Identify</em>, browse the full specimen index, or
    review how the model was built and validated.</div>
    """, unsafe_allow_html=True)

    divider()

    if eval_summary and class_mapping:
        c1, c2, c3, c4 = st.columns(4)
        specimen_stat_card(c1, "FIELD NO. 001", "Test Accuracy", "Accuratio testis", f"{eval_summary['test_accuracy']*100:.1f}", "%")
        specimen_stat_card(c2, "FIELD NO. 002", "Macro F1", "Media harmonica", f"{eval_summary['macro_f1']:.3f}")
        specimen_stat_card(c3, "FIELD NO. 003", "Catalogued Species", "Species catalogatae", f"{len(class_mapping)}")
        specimen_stat_card(c4, "FIELD NO. 004", "Specimens Examined", "Specimina inspecta", f"{eval_summary['n_test_samples']:,}")
    else:
        field_note("Evaluation Pending", "Run <code>evaluate.py</code> (Phase 6) to populate these figures.", "warn")

    divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        section("Collection Overview", "🌿")
        if class_dist:
            crops = Counter(row["class_name"].split("___")[0] for row in class_dist)
            fig = px.bar(
                x=list(crops.keys()), y=list(crops.values()),
                title="Catalogued Conditions per Crop Species",
                labels={"x": "Crop", "y": "Conditions"},
            )
            fig.update_traces(marker_color="#4F7A5E", marker_line_width=0)
            st.plotly_chart(herb_chart(fig), use_container_width=True)
        else:
            field_note("Index Not Built", "Run <code>data_split.py</code> and the EDA notebook to populate the collection index.", "warn")

    with col2:
        section("Field Log", "🍃")
        field_note("Baseline", "Simple 4-block CNN. <strong>96.4%</strong> test accuracy — the reference specimen.", "")
        field_note("Improved", "BatchNorm + GAP + deeper capacity. <strong>97.6%</strong> test accuracy, "
                    "train/val gap closed from 0.036 to near-zero.", "")
        field_note("Tuning", "Skipped deliberately — diminishing returns given CPU training cost and "
                    "an already near-ceiling result.", "info")

    caveat_banner()


# ============================================================================
# PAGE: IDENTIFY
# ============================================================================

def page_identify():
    st.markdown('<div class="field-label">🔍 Live Identification</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Identify a specimen</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a leaf photograph for classification and an attention overlay.</div>', unsafe_allow_html=True)

    caveat_banner()

    ckpt_path = MODELS_DIR / "improved" / "improved_best.pt"
    if not ckpt_path.exists():
        field_note("Model Not Found", f"No checkpoint at <code>{ckpt_path}</code>. Complete Phase 4 training first.", "danger")
        return

    model, idx_to_class, device, gradcam = get_model_and_gradcam()

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = predict_image(model, idx_to_class, device, image, top_k=3)
        top_class, top_confidence = results[0]
        crop, disease = parse_class_name(top_class)

        col1, col2 = st.columns(2)
        with col1:
            section("Specimen Photograph", "📷")
            st.image(image, use_container_width=True)
        with col2:
            section("Model Attention", "🔥")
            transform = get_eval_transforms()
            input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            input_tensor.requires_grad_()
            cam, _, _ = gradcam.generate(input_tensor)
            image_np = denormalize_image(input_tensor.squeeze(0))
            overlay = overlay_heatmap(image_np, cam)
            st.image(overlay, use_container_width=True, clamp=True)

        divider()

        st.markdown(f"""
        <div class="specimen-card" style="max-width:520px;">
            <div class="specimen-field-no">IDENTIFICATION · {top_class}</div>
            <div class="specimen-title">{crop}</div>
            <div class="specimen-latin">{disease}</div>
            <div class="specimen-stat-row">
                <span class="specimen-stat-label">Confidence</span>
                <span class="specimen-stat-val">{top_confidence*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(top_confidence)

        if top_confidence < 0.6:
            field_note("Low Confidence", "The model is uncertain — the image may be ambiguous, "
                       "poorly lit, or outside the training distribution.", "warn")

        with st.expander("Alternate identifications"):
            for class_name, prob in results:
                c, d = parse_class_name(class_name)
                st.write(f"**{c} — {d}**: {prob*100:.1f}%")
    else:
        field_note("Awaiting Specimen", "Upload an image above to begin identification.", "info")


# ============================================================================
# PAGE: SPECIMEN INDEX (dataset explorer)
# ============================================================================

def page_specimen_index(class_dist, class_mapping):
    st.markdown('<div class="field-label">🗂️ Collection Catalogue</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Specimen index</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">The full PlantVillage (color) catalogue — 38 conditions, browsable by species.</div>', unsafe_allow_html=True)

    if not class_dist:
        field_note("Index Not Available", "Run <code>data_split.py</code> first to generate the catalogue.", "warn")
        return

    divider()

    tab1, tab2, tab3 = st.tabs(["Distribution", "Browse Specimens", "Imbalance Notes"])

    with tab1:
        section("Specimens per Class", "🌿")
        sorted_dist = sorted(class_dist, key=lambda r: int(r["total"]), reverse=True)
        fig = px.bar(
            x=[int(r["total"]) for r in sorted_dist],
            y=[r["class_name"] for r in sorted_dist],
            orientation="h",
            title="Image Count per Catalogued Condition",
            labels={"x": "Images", "y": ""},
        )
        fig.update_traces(marker_color="#4F7A5E", marker_line_width=0)
        fig.update_layout(height=850, yaxis=dict(autorange="reversed"), yaxis_tickfont=dict(size=9))
        st.plotly_chart(herb_chart(fig), use_container_width=True)

    with tab2:
        section("Browse by Species", "🌾")
        if SAMPLES_DIR.exists():
            class_names = sorted(class_mapping.keys()) if class_mapping else sorted(r["class_name"] for r in class_dist)
            selected_class = st.selectbox("Select a catalogued condition", class_names)

            class_dir = SAMPLES_DIR / selected_class
            if class_dir.exists():
                images = sorted(class_dir.glob("*.[jJ][pP][gG]"))[:6] + sorted(class_dir.glob("*.[pP][nN][gG]"))[:6]
                images = images[:6]
                crop, disease = parse_class_name(selected_class)

                cols = st.columns(min(6, max(1, len(images))))
                for i, img_path in enumerate(images):
                    with cols[i % len(cols)]:
                        st.image(str(img_path), use_container_width=True)
                        st.markdown(f"""
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:var(--ink-faint);text-align:center;margin-top:-8px;">
                            SPEC. {i+1:03d}
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="specimen-card" style="margin-top:1rem;max-width:420px;">
                    <div class="specimen-field-no">{selected_class}</div>
                    <div class="specimen-title">{crop}</div>
                    <div class="specimen-latin">{disease}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                field_note("Folder Not Found", f"Expected images at <code>{class_dir}</code>.", "warn")
        else:
            field_note("Samples Not Found", f"Expected sample thumbnails at <code>{SAMPLES_DIR}</code>. Run <code>build_specimen_samples.py</code> first.", "warn")

    with tab3:
        section("Class Imbalance", "⚖️")
        sorted_dist = sorted(class_dist, key=lambda r: int(r["total"]), reverse=True)
        largest, smallest = sorted_dist[0], sorted_dist[-1]
        ratio = int(largest["total"]) / int(smallest["total"])

        c1, c2, c3 = st.columns(3)
        specimen_stat_card(c1, "LARGEST", largest["class_name"].split("___")[0], parse_class_name(largest["class_name"])[1], largest["total"])
        specimen_stat_card(c2, "SMALLEST", smallest["class_name"].split("___")[0], parse_class_name(smallest["class_name"])[1], smallest["total"])
        specimen_stat_card(c3, "IMBALANCE RATIO", "Largest : Smallest", "Ratio observata", f"{ratio:.0f}", " : 1")

        divider()
        field_note("How This Was Handled", "A <strong>WeightedRandomSampler</strong> rebalances the training "
                   "batches so every class gets roughly equal exposure per epoch, regardless of raw sample count. "
                   "Validation and test sets are left at their natural distribution.", "info")


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

def page_performance(eval_summary, report_text):
    st.markdown('<div class="field-label">📊 Validation Record</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Model performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Baseline vs. improved architecture, and full test-set evaluation.</div>', unsafe_allow_html=True)

    divider()

    tab1, tab2, tab3 = st.tabs(["Baseline vs Improved", "Confusion Matrix", "Classification Report"])

    with tab1:
        section("Training Curves", "📈")
        baseline_hist = load_history("baseline")
        improved_hist = load_history("improved")

        if baseline_hist and improved_hist:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=baseline_hist["val_loss"], name="Baseline", line=dict(dash="dot", color="#A3492F")))
                fig.add_trace(go.Scatter(y=improved_hist["val_loss"], name="Improved", line=dict(color="#4F7A5E")))
                fig.update_layout(title="Validation Loss")
                st.plotly_chart(herb_chart(fig), use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=baseline_hist["val_acc"], name="Baseline", line=dict(dash="dot", color="#A3492F")))
                fig2.add_trace(go.Scatter(y=improved_hist["val_acc"], name="Improved", line=dict(color="#4F7A5E")))
                fig2.update_layout(title="Validation Accuracy")
                st.plotly_chart(herb_chart(fig2), use_container_width=True)

            gap_baseline = baseline_hist["val_acc"][-1] - baseline_hist["train_acc"][-1]
            gap_improved = improved_hist["val_acc"][-1] - improved_hist["train_acc"][-1]
            field_note("Train/Val Gap", f"Baseline: <strong>{gap_baseline:+.4f}</strong> &nbsp;|&nbsp; "
                       f"Improved: <strong>{gap_improved:+.4f}</strong> — a gap near zero indicates the "
                       "model generalizes rather than memorizes.", "")
        else:
            field_note("Training Logs Not Found", "Run Phase 3 and Phase 4 training first.", "warn")

        curves_path = PLOTS_DIR / "baseline_vs_improved_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), use_container_width=True)

    with tab2:
        section("Confusion Matrix", "🧩")
        cm_path = PLOTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
            field_note("How to Read This", "Confusion concentrated near the diagonal within the same crop's "
                       "block is expected — different diseases on the same species often look visually similar.", "info")
        else:
            field_note("Not Generated Yet", "Run <code>evaluate.py</code> (Phase 6) to produce the confusion matrix.", "warn")

    with tab3:
        section("Per-Class Report", "📋")
        if eval_summary:
            c1, c2 = st.columns(2)
            specimen_stat_card(c1, "SUMMARY", "Macro F1", "Media inter species", f"{eval_summary['macro_f1']:.4f}")
            specimen_stat_card(c2, "SUMMARY", "Weighted F1", "Media ponderata", f"{eval_summary['weighted_f1']:.4f}")
            divider()
        if report_text:
            st.code(report_text, language="text")
        else:
            field_note("Report Not Found", "Run <code>evaluate.py</code> (Phase 6) to generate the classification report.", "warn")


# ============================================================================
# PAGE: EXPLAINABILITY
# ============================================================================

def page_explainability():
    st.markdown('<div class="field-label">👁️ Attention Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Grad-CAM overlays — what the model attends to when identifying a specimen.</div>', unsafe_allow_html=True)

    divider()

    if not GRADCAM_DIR.exists() or not any(GRADCAM_DIR.iterdir()):
        field_note("Gallery Empty", "Run <code>gradcam.py</code> (Phase 7) to populate this gallery.", "warn")
        return

    tab1, tab2 = st.tabs(["Correct Identifications", "Incorrect Identifications"])

    with tab1:
        section("Correctly Identified Specimens", "✅")
        correct_images = sorted(GRADCAM_DIR.glob("correct_*.png"))
        if correct_images:
            cols = st.columns(2)
            for i, img_path in enumerate(correct_images):
                with cols[i % 2]:
                    st.markdown(f'<div class="specimen-field-no">SPEC. {i+1:03d} — CORRECT</div>', unsafe_allow_html=True)
                    st.image(str(img_path), use_container_width=True)
        else:
            field_note("None Found", "No correct-prediction Grad-CAM samples generated.", "info")

    with tab2:
        section("Misidentified Specimens", "🍁")
        incorrect_images = sorted(GRADCAM_DIR.glob("incorrect_*.png"))
        if incorrect_images:
            cols = st.columns(2)
            for i, img_path in enumerate(incorrect_images):
                with cols[i % 2]:
                    st.markdown(f'<div class="specimen-field-no">SPEC. {i+1:03d} — INCORRECT</div>', unsafe_allow_html=True)
                    st.image(str(img_path), use_container_width=True)
            field_note("What To Look For", "Check whether the heatmap lands on a plausible-but-wrong "
                       "symptom, versus somewhere irrelevant like the background — the latter would "
                       "suggest a shortcut rather than a genuine disease feature.", "info")
        else:
            field_note("None Found", "No incorrect predictions were sampled — consistent with the model's "
                       "high test accuracy.", "")


# ============================================================================
# PAGE: FIELD NOTES (error analysis)
# ============================================================================

def page_field_notes(test_predictions):
    st.markdown('<div class="field-label">🍂 Error Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Field notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Where and why the model gets it wrong.</div>', unsafe_allow_html=True)

    divider()

    if not test_predictions:
        field_note("No Predictions Found", "Run <code>evaluate.py</code> (Phase 6) first.", "warn")
        return

    misclassified = [r for r in test_predictions if r["correct"] == "0"]
    total = len(test_predictions)

    c1, c2, c3 = st.columns(3)
    specimen_stat_card(c1, "TOTAL", "Specimens Tested", "Specimina probata", f"{total:,}")
    specimen_stat_card(c2, "ERRORS", "Misidentified", "Errores", f"{len(misclassified)}")
    specimen_stat_card(c3, "RATE", "Error Rate", "Ratio errorum", f"{len(misclassified)/total*100:.1f}", "%")

    divider()

    tab1, tab2, tab3 = st.tabs(["Confused Pairs", "Confidence Breakdown", "Misidentified Gallery"])

    with tab1:
        section("Most Confused Pairs", "🔀")
        pair_counts = Counter((r["true_class"], r["pred_class"]) for r in misclassified)
        if pair_counts:
            top_pairs = pair_counts.most_common(15)
            fig = px.bar(
                x=[c for _, c in top_pairs],
                y=[f"{t} → {p}" for (t, p), _ in top_pairs],
                orientation="h",
                title="True → Predicted (top 15)",
                labels={"x": "Count", "y": ""},
            )
            fig.update_traces(marker_color="#A3492F", marker_line_width=0)
            fig.update_layout(height=520, yaxis=dict(autorange="reversed"), yaxis_tickfont=dict(size=9))
            st.plotly_chart(herb_chart(fig), use_container_width=True)

            def crop_of(c): return c.split("___")[0]
            within = sum(1 for r in misclassified if crop_of(r["true_class"]) == crop_of(r["pred_class"]))
            cross = len(misclassified) - within
            field_note("Within-Crop vs Cross-Crop",
                       f"<strong>{within}</strong> errors within the same crop (expected — similar-looking "
                       f"symptoms) vs <strong>{cross}</strong> cross-crop errors "
                       f"{'(worth investigating — possible shortcut learning)' if cross > 0 else '(none — a good sign)'}.",
                       "info" if cross == 0 else "warn")
        else:
            field_note("No Errors Found", "The model made no mistakes on the test set.", "")

    with tab2:
        section("Confidence of Mistakes", "📉")
        if misclassified:
            confidences = [float(r["confidence"]) for r in misclassified]
            fig = px.histogram(x=confidences, nbins=20, title="Confidence Distribution — Misclassified Only",
                               labels={"x": "Confidence", "y": "Count"})
            fig.update_traces(marker_color="#B4863B", marker_line_width=0)
            st.plotly_chart(herb_chart(fig), use_container_width=True)

            high_conf_wrong = [r for r in misclassified if float(r["confidence"]) >= 0.9]
            if high_conf_wrong:
                field_note("High-Confidence Errors",
                           f"<strong>{len(high_conf_wrong)}</strong> errors where the model was ≥90% confident "
                           "and still wrong — these are worth individually reviewing; they may indicate a "
                           "source labeling issue rather than a model weakness.", "danger")
            else:
                field_note("No High-Confidence Errors", "Every mistake had under 90% confidence — the model "
                           "was appropriately uncertain when wrong.", "")

    with tab3:
        section("Misidentified Specimens", "🍁")
        if MISCLASSIFIED_DIR.exists() and any(MISCLASSIFIED_DIR.iterdir()):
            images = sorted(MISCLASSIFIED_DIR.glob("*.png"))
            cols = st.columns(2)
            for i, img_path in enumerate(images):
                with cols[i % 2]:
                    st.image(str(img_path), use_container_width=True)
        else:
            field_note("Gallery Empty", "Run the error-analysis notebook (Phase 8) to populate this gallery.", "warn")


# ============================================================================
# PAGE: ABOUT
# ============================================================================

def page_about(class_mapping):
    st.markdown('<div class="field-label">🌱 Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">About this atlas</div>', unsafe_allow_html=True)

    divider()
    section("Pipeline", "🌱")

    phases = [
        ("01", "EDA", "Class distribution, image consistency, duplicate detection."),
        ("02", "Data Processing", "Stratified 70/15/15 split, weighted sampling for imbalance."),
        ("03", "Baseline CNN", "4-block scratch CNN — 96.4% test accuracy."),
        ("04", "Improved CNN", "BatchNorm + GAP + deeper capacity — 97.6% test accuracy, near-zero train/val gap."),
        ("05", "Tuning", "Deliberately skipped — diminishing returns given CPU cost and near-ceiling result."),
        ("06", "Evaluation", "Macro-F1, confusion matrix, per-class precision/recall."),
        ("07", "Explainability", "Grad-CAM on correct and incorrect predictions."),
        ("08", "Error Analysis", "Confused-pair inspection, confidence breakdown."),
        ("09", "Deployment", "This application."),
    ]

    for num, title, desc in phases:
        st.markdown(f"""
        <div class="specimen-card" style="margin-bottom:0.6rem;">
            <div style="display:flex;gap:1rem;align-items:baseline;">
                <span style="font-family:'IBM Plex Mono',monospace;color:var(--moss);font-weight:600;">{num}</span>
                <span class="specimen-title" style="margin:0;">{title}</span>
            </div>
            <div class="specimen-latin" style="margin-top:0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    divider()
    section("Dataset", "🍂")
    n_classes = len(class_mapping) if class_mapping else "38"
    field_note("PlantVillage (Color)", f"{n_classes} catalogued conditions across 14 crop species, "
               "photographed against plain backgrounds under controlled conditions.", "")
    caveat_banner()


# ============================================================================
# MAIN
# ============================================================================

def main():
    class_mapping = load_class_mapping()
    class_dist = load_class_distribution()
    eval_summary = load_evaluation_summary()
    test_predictions = load_test_predictions()
    report_text = load_classification_report_text()

    with st.sidebar:
        st.markdown("""
        <div class="nav-logo">
            <div class="nav-logo-mark">The Herbarium</div>
            <div class="nav-logo-sub">Botanical Diagnostic Atlas</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-section-title">Field Log</div>', unsafe_allow_html=True)

        page_options = {
            "🌿  Home":              "home",
            "🔍  Identify":          "identify",
            "🌾  Specimen Index":    "index",
            "📊  Model Performance": "performance",
            "👁️  Explainability":    "explain",
            "🍂  Field Notes":       "notes",
            "🌱  About":             "about",
        }
        selected = st.radio("nav", list(page_options.keys()), label_visibility="collapsed")
        page_key = page_options[selected]

        st.divider()
        st.markdown('<div class="nav-section-title">Collection Snapshot</div>', unsafe_allow_html=True)
        if class_mapping:
            st.markdown(f'<div class="sq-stat"><span class="sq-label">Species Catalogued</span><span class="sq-value">{len(class_mapping)}</span></div>', unsafe_allow_html=True)
        if eval_summary:
            st.markdown(f'<div class="sq-stat"><span class="sq-label">Test Accuracy</span><span class="sq-value">{eval_summary["test_accuracy"]*100:.1f}%</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sq-stat"><span class="sq-label">Macro F1</span><span class="sq-value">{eval_summary["macro_f1"]:.3f}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="padding: 0 0.5rem; color: #5A6E5F; font-size: 0.74rem; line-height: 1.6;">
            PlantVillage (Color) · PyTorch<br>
            <span style="opacity:0.7;">Herbarium Atlas · 2026</span>
        </div>
        """, unsafe_allow_html=True)

    if   page_key == "home":        page_home(class_mapping, class_dist, eval_summary)
    elif page_key == "identify":    page_identify()
    elif page_key == "index":       page_specimen_index(class_dist, class_mapping)
    elif page_key == "performance": page_performance(eval_summary, report_text)
    elif page_key == "explain":     page_explainability()
    elif page_key == "notes":       page_field_notes(test_predictions)
    elif page_key == "about":       page_about(class_mapping)


if __name__ == "__main__":
    main()