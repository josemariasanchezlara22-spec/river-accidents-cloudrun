import io
import html
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from river import preprocessing, tree, metrics


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RoadRisk AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --surface: #ffffff;
        --surface-soft: #f9fbfe;
        --surface-tint: #edf3ff;
        --primary: #0f3b66;
        --primary-2: #1756a9;
        --accent: #2563eb;
        --accent-2: #0ea5e9;
        --text: #132238;
        --text-soft: #52637a;
        --muted: #718096;
        --border: #d8e2ee;
        --shadow: 0 14px 34px rgba(16, 42, 67, 0.08);
        --shadow-soft: 0 8px 24px rgba(16, 42, 67, 0.06);
        --success: #127b4f;
        --success-bg: #e8f7ef;
        --danger: #b42318;
        --danger-bg: #fdecec;
        --warning: #b45309;
        --warning-bg: #fff1dc;
        --info-bg: #dbeafe;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg) 28%, #f4f7fb 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .page-head {
        background: linear-gradient(135deg, #0f3b66 0%, #1756a9 58%, #2563eb 100%);
        color: #ffffff;
        padding: 1.45rem 1.5rem;
        border-radius: 22px;
        box-shadow: 0 18px 40px rgba(15, 59, 102, 0.20);
        margin-bottom: 1rem;
    }

    .page-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.26);
        color: #ffffff !important;
        border-radius: 999px;
        padding: 0.34rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .page-head h1 {
        margin: 0;
        color: #ffffff !important;
        font-size: 2.05rem;
        font-weight: 900;
    }

    .page-head p {
        margin: 0.45rem 0 0;
        color: rgba(255, 255, 255, 0.86) !important;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 980px;
    }

    .surface {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
        color: var(--text);
    }

    .card-title {
        font-size: 1rem;
        font-weight: 900;
        color: var(--primary) !important;
        margin-bottom: 0.25rem;
    }

    .card-subtitle {
        color: var(--muted) !important;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.95rem;
    }

    .small-code {
        background: #f7faff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.85rem;
        font-family: monospace;
        font-size: 0.86rem;
        color: #1e3a5f !important;
        word-break: break-all;
    }

    .surface-head {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        margin-bottom: 0.95rem;
    }

    .surface-title {
        font-size: 1rem;
        font-weight: 900;
        color: var(--primary) !important;
    }

    .surface-subtitle {
        color: var(--muted) !important;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .section-title {
        color: var(--text) !important;
        font-size: 1.05rem;
        font-weight: 900;
        margin: 0.9rem 0 0.65rem;
        letter-spacing: 0;
    }

    .section-note {
        color: var(--muted) !important;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
    }

    .status-banner {
        border-radius: 16px;
        padding: 0.95rem 1rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .status-ok {
        background: var(--success-bg);
        color: var(--success) !important;
        border-color: #bfead1;
    }

    .status-good {
        background: var(--success-bg);
        color: var(--success) !important;
        border-color: #bfead1;
    }

    .status-neutral {
        background: var(--info-bg);
        color: var(--primary-2) !important;
        border-color: #93c5fd;
    }

    .status-warn {
        background: var(--warning-bg);
        color: var(--warning) !important;
        border-color: #f2ce88;
    }

    .status-info {
        background: var(--info-bg);
        color: var(--primary-2) !important;
        border-color: #93c5fd;
    }

    .status-danger {
        background: var(--danger-bg);
        color: var(--danger) !important;
        border-color: #f5b5b0;
    }

    .risk-band {
        border-radius: 18px;
        padding: 1.15rem 1.15rem;
        font-weight: 900;
        font-size: 1.15rem;
        text-align: center;
        border: 1px solid transparent;
        box-shadow: var(--shadow-soft);
    }

    .risk-high {
        background: #fff0ef;
        border-color: #f5b5b0;
        color: var(--danger) !important;
    }

    .risk-low {
        background: var(--success-bg);
        border-color: #bfead1;
        color: var(--success) !important;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .info-item {
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
    }

    .info-label {
        color: var(--muted) !important;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.18rem;
    }

    .info-value {
        color: var(--text) !important;
        font-size: 0.96rem;
        font-weight: 700;
        line-height: 1.45;
        word-break: break-word;
    }

    .artifact-path {
        background: #f7faff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.72rem 0.8rem;
        font-family: monospace;
        font-size: 0.86rem;
        color: #1e3a5f !important;
        word-break: break-all;
    }

    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.35rem;
        box-shadow: var(--shadow-soft);
        gap: 0.3rem;
    }

    div[role="radiogroup"] label {
        background: transparent;
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        margin-right: 0;
        color: var(--text) !important;
        font-weight: 800;
        border: 1px solid transparent;
    }

    div[role="radiogroup"] label p {
        color: var(--text) !important;
        font-weight: 800;
    }

    div[role="radiogroup"] label:hover {
        background: #eaf2ff;
        border-color: #c7dbff;
    }

    div[role="radiogroup"] input:checked + div {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: var(--shadow-soft);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 800;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 800;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        color: var(--primary) !important;
        font-weight: 900;
    }

    div[data-testid="stMetricValue"] div {
        color: var(--primary) !important;
    }

    /* ---------------- ALERTS ---------------- */

    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid #b9d7ff;
        background-color: #dbeafe;
        color: #0b2545 !important;
    }

    div[data-testid="stAlert"] * {
        color: #0b2545 !important;
    }

    /* ---------------- INPUTS ---------------- */

    input, textarea, select {
        color: var(--text) !important;
        background-color: #ffffff !important;
        border-color: var(--border) !important;
    }

    label, label p {
        color: var(--text-soft) !important;
        font-weight: 700;
    }

    /* ---------------- DATAFRAME / TABLES ---------------- */

    .stDataFrame {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid var(--border);
    }

    /* ---------------- FOOTER ---------------- */

    footer {
        visibility: hidden;
    }

    .app-footer {
        margin-top: 2rem;
        padding: 1rem 0;
        color: var(--muted) !important;
        font-size: 0.85rem;
        border-top: 1px solid var(--border);
    }

    .app-footer span {
        color: var(--muted) !important;
    }

    /* ---------------- BUTTONS ---------------- */

    .stButton > button {
        border-radius: 999px;
        font-weight: 800;
        border: 1px solid var(--accent);
        padding-left: 1rem;
        padding-right: 1rem;
        box-shadow: none;
        background: #ffffff;
        color: var(--primary) !important;
        min-height: 2.75rem;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #ffffff !important;
        border-color: var(--accent);
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: var(--primary) !important;
        border-color: #9bb7da;
    }

    .stButton > button[data-testid="baseButton-secondary"] {
        background: #ffffff;
        color: var(--primary) !important;
        border-color: #9bb7da;
    }

    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--accent);
        color: #ffffff !important;
        border-color: var(--accent);
    }

    .stButton > button:disabled {
        background: #eef2f7 !important;
        color: #94a3b8 !important;
        border-color: #d5deea !important;
        cursor: not-allowed;
        opacity: 1;
        box-shadow: none;
        transform: none;
    }

    .stButton > button:hover {
        border-color: var(--primary-2);
        color: var(--primary-2) !important;
        background: #f8fbff;
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(16, 42, 67, 0.08);
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: #1f5fe0;
        color: #ffffff !important;
        border-color: #1f5fe0;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #f8fbff;
        color: var(--primary-2) !important;
        border-color: #8fb0d6;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label * {
        color: var(--text) !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] {
        color: var(--text) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    .surface h1, .surface h2, .surface h3, .surface h4, .surface h5, .surface h6,
    .card h1, .card h2, .card h3, .card h4, .card h5, .card h6 {
        color: var(--text) !important;
    }

    code {
        color: #174ea6 !important;
        background: #eef5ff !important;
        border-radius: 6px;
        padding: 0.1rem 0.3rem;
    }

    pre, pre code {
        color: #d1fae5 !important;
        background: #0f172a !important;
    }

    .mono-block {
        background: #f7faff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
        font-family: monospace;
        font-size: 0.86rem;
        color: #1e3a5f !important;
        word-break: break-all;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CONFIG
# =========================================================

DEFAULT_PROJECT_ID = os.getenv("PROJECT_ID", "ml-big-data-q2-up")
DEFAULT_BUCKET_NAME = os.getenv("BUCKET_NAME", "us-accidents-am-up-02")
DEFAULT_PREFIX = os.getenv("DATA_PREFIX", "raw/")

MODEL_PATH = os.getenv("MODEL_PATH", "models/modelo_incremental_ht.pkl")
HISTORY_PATH = os.getenv("HISTORY_PATH", "models/history_incremental.csv")

TARGET_ORIGINAL = "Severity"
TARGET_BINARY = "is_severe"
TIME_COL = "Start_Time"

NUM_FEATURES = [
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)"
]

CAT_FEATURES = [
    "State",
    "Weather_Condition",
    "year_month"
]

BOOL_FEATURES = [
    "Amenity",
    "Crossing",
    "Junction",
    "Traffic_Signal"
]

DERIVED_TIME_FEATURES = [
    "hour",
    "month",
    "dayofweek",
    "is_weekend"
]

SELECTED_FEATURES = (
    NUM_FEATURES
    + CAT_FEATURES
    + BOOL_FEATURES
    + DERIVED_TIME_FEATURES
)


# =========================================================
# UI HELPERS
# =========================================================


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def render_page_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="page-head">
            <div class="page-badge">{esc(eyebrow)}</div>
            <h1>{esc(title)}</h1>
            <p>{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_status_banner(message: str, tone: str = "info"):
    st.markdown(
        f'<div class="status-banner status-{tone}">{esc(message)}</div>',
        unsafe_allow_html=True
    )


def render_surface(title: str, subtitle: str = ""):
    st.markdown('<section class="surface">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="surface-head">
            <div class="surface-title">{esc(title)}</div>
            {"<div class='surface-subtitle'>" + esc(subtitle) + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True
    )


def close_surface():
    st.markdown('</section>', unsafe_allow_html=True)


def render_path_block(path: str):
    st.markdown(f'<div class="artifact-path">{esc(path)}</div>', unsafe_allow_html=True)


def render_info_grid(items):
    st.markdown('<div class="info-grid">', unsafe_allow_html=True)
    for label, value in items:
        st.markdown(
            f"""
            <div class="info-item">
                <div class="info-label">{esc(label)}</div>
                <div class="info-value">{esc(value)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# GCS
# =========================================================

@st.cache_resource
def get_storage_client():
    return storage.Client()


def get_bucket(bucket_name: str):
    return get_storage_client().bucket(bucket_name)


def load_pickle_from_gcs(bucket_name: str, blob_name: str):
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    data = blob.download_as_bytes()
    return pickle.loads(data)


def save_pickle_to_gcs(obj, bucket_name: str, blob_name: str):
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pickle.dumps(obj))


def load_history_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return pd.DataFrame()

    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def save_history_to_gcs(df: pd.DataFrame, bucket_name: str, blob_name: str):
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv"
    )


def delete_blob_if_exists(bucket_name: str, blob_name: str):
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if blob.exists():
        blob.delete()


def list_csv_files(bucket_name: str, prefix: str):
    bucket = get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    return sorted([
        blob.name
        for blob in blobs
        if blob.name.lower().endswith(".csv")
    ])


def read_csv_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


# =========================================================
# MODEL
# =========================================================

def new_model_bundle():
    return {
        "encoder": preprocessing.OneHotEncoder(),
        "classifier": tree.HoeffdingTreeClassifier(),
        "model_name": "HoeffdingTreeClassifier",
        "created_at": datetime.utcnow().isoformat(),
        "rows_trained": 0,
        "last_training_at": None
    }


def transform_x(model_bundle, x: dict):
    return model_bundle["encoder"].transform_one(x)


def learn_transformers(model_bundle, x: dict):
    model_bundle["encoder"].learn_one(x)
    return model_bundle["encoder"].transform_one(x)


def predict_one(model_bundle, x: dict):
    x_encoded = transform_x(model_bundle, x)

    y_pred = model_bundle["classifier"].predict_one(x_encoded)

    if y_pred is None:
        y_pred = 0

    proba = model_bundle["classifier"].predict_proba_one(x_encoded)

    return int(y_pred), proba


def learn_one(model_bundle, x: dict, y: int):
    x_encoded = learn_transformers(model_bundle, x)

    model_bundle["classifier"].learn_one(x_encoded, y)
    model_bundle["rows_trained"] = model_bundle.get("rows_trained", 0) + 1
    model_bundle["last_training_at"] = datetime.utcnow().isoformat()

    return model_bundle


def make_metric_bundle():
    return {
        "accuracy": metrics.Accuracy(),
        "precision": metrics.Precision(),
        "recall": metrics.Recall(),
        "f1": metrics.F1()
    }


def update_metric_bundle(metric_bundle, y_true, y_pred):
    for metric in metric_bundle.values():
        metric.update(y_true, y_pred)


def metric_snapshot(metric_bundle):
    return {
        name: metric.get()
        for name, metric in metric_bundle.items()
    }


# =========================================================
# DATA PREP
# =========================================================

def prepare_accidents_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_ORIGINAL not in df.columns:
        raise ValueError(f"El archivo no contiene la columna {TARGET_ORIGINAL}")

    df = df[df[TARGET_ORIGINAL].notna()].copy()

    df[TARGET_ORIGINAL] = pd.to_numeric(
        df[TARGET_ORIGINAL],
        errors="coerce"
    )

    df = df[df[TARGET_ORIGINAL].notna()].copy()

    df[TARGET_BINARY] = (df[TARGET_ORIGINAL] >= 3).astype(int)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(
            df[TIME_COL],
            errors="coerce"
        )

        df["hour"] = df[TIME_COL].dt.hour
        df["month"] = df[TIME_COL].dt.month
        df["dayofweek"] = df[TIME_COL].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        if "year_month" not in df.columns:
            df["year_month"] = df[TIME_COL].dt.to_period("M").astype(str)

    for col in NUM_FEATURES + DERIVED_TIME_FEATURES:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce")

        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(median_value)

    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].fillna("Unknown").astype(str)

    for col in BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False

        df[col] = df[col].fillna(False).astype(int)

    if TIME_COL in df.columns:
        df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    return df[SELECTED_FEATURES + [TARGET_BINARY, TARGET_ORIGINAL]]


def row_to_x(row) -> dict:
    x = {}

    for feature in SELECTED_FEATURES:
        value = row[feature]

        if pd.isna(value):
            value = "Unknown" if feature in CAT_FEATURES else 0

        if feature in CAT_FEATURES:
            x[feature] = str(value)
        else:
            x[feature] = float(value)

    return x


def build_manual_x(values: dict):
    return {
        "Distance(mi)": float(values["distance"]),
        "Temperature(F)": float(values["temperature"]),
        "Humidity(%)": float(values["humidity"]),
        "Visibility(mi)": float(values["visibility"]),
        "Wind_Speed(mph)": float(values["wind_speed"]),
        "State": str(values["state"]),
        "Weather_Condition": str(values["weather"]),
        "year_month": str(values["year_month"]),
        "Amenity": float(values["amenity"]),
        "Crossing": float(values["crossing"]),
        "Junction": float(values["junction"]),
        "Traffic_Signal": float(values["traffic_signal"]),
        "hour": float(values["hour"]),
        "month": float(values["month"]),
        "dayofweek": float(values["dayofweek"]),
        "is_weekend": float(values["is_weekend"]),
    }


# =========================================================
# TRAINING
# =========================================================

def train_on_file(
    model_bundle,
    bucket_name: str,
    blob_name: str,
    max_rows: int,
    balance_training: bool,
    random_state: int = 42
):
    df_raw = read_csv_from_gcs(bucket_name, blob_name)
    df = prepare_accidents_df(df_raw)

    if len(df) == 0:
        raise ValueError("El archivo no tiene registros validos despues de preparar datos")

    # En flujo incremental preferimos procesar una ventana temporal contigua
    # en lugar de una muestra aleatoria, porque la aleatoriedad rompe la secuencia.
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()

    class_counts = df[TARGET_BINARY].value_counts()
    n_0 = int(class_counts.get(0, 0))
    n_1 = int(class_counts.get(1, 0))

    if n_0 > 0 and n_1 > 0:
        keep_prob_majority = min(1.0, n_1 / n_0)
    else:
        keep_prob_majority = 1.0

    rng = np.random.default_rng(random_state)
    metric_bundle = make_metric_bundle()

    learned_rows = 0
    evaluated_rows = 0
    start = time.time()

    for _, row in df.iterrows():
        y = int(row[TARGET_BINARY])
        x = row_to_x(row)

        y_pred, _ = predict_one(model_bundle, x)
        update_metric_bundle(metric_bundle, y, y_pred)

        should_learn = True

        if balance_training and y == 0:
            if rng.random() > keep_prob_majority:
                should_learn = False

        if should_learn:
            learn_one(model_bundle, x, y)
            learned_rows += 1

        evaluated_rows += 1

    result = metric_snapshot(metric_bundle)

    result.update({
        "file": blob_name.split("/")[-1],
        "blob_name": blob_name,
        "rows_file": len(df),
        "rows_evaluated": evaluated_rows,
        "rows_learned": learned_rows,
        "severe_rate_file": float(df[TARGET_BINARY].mean()),
        "balance_training": balance_training,
        "keep_prob_majority": keep_prob_majority,
        "elapsed_sec": round(time.time() - start, 2),
        "processed_at": datetime.utcnow().isoformat(),
        "model_rows_trained_total": model_bundle.get("rows_trained", 0)
    })

    return model_bundle, result


# =========================================================
# STATE INIT
# =========================================================

if "project_id" not in st.session_state:
    st.session_state.project_id = DEFAULT_PROJECT_ID

if "bucket_name" not in st.session_state:
    st.session_state.bucket_name = DEFAULT_BUCKET_NAME

if "prefix" not in st.session_state:
    st.session_state.prefix = DEFAULT_PREFIX

if "max_rows" not in st.session_state:
    st.session_state.max_rows = 8000

if "balance_training" not in st.session_state:
    st.session_state.balance_training = True

bucket_name = st.session_state.bucket_name
prefix = st.session_state.prefix

if "model_bundle" not in st.session_state:
    try:
        loaded_model = load_pickle_from_gcs(bucket_name, MODEL_PATH)

        if loaded_model is None:
            st.session_state.model_bundle = new_model_bundle()
            st.session_state.model_status = "Nuevo modelo creado"
            st.session_state.model_status_type = "warn"
        else:
            st.session_state.model_bundle = loaded_model
            st.session_state.model_status = "Modelo cargado desde GCS"
            st.session_state.model_status_type = "good"

    except Exception as e:
        st.session_state.model_bundle = new_model_bundle()
        st.session_state.model_status = f"No se pudo cargar el modelo: {e}"
        st.session_state.model_status_type = "warn"

if "history" not in st.session_state:
    try:
        st.session_state.history = load_history_from_gcs(bucket_name, HISTORY_PATH)
    except Exception:
        st.session_state.history = pd.DataFrame()

if "files" not in st.session_state:
    st.session_state.files = []

if "file_index" not in st.session_state:
    st.session_state.file_index = 0

if "last_x" not in st.session_state:
    st.session_state.last_x = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


model_bundle = st.session_state.model_bundle
history = st.session_state.history


# =========================================================
# HEADER
# =========================================================

render_page_header(
    "Online learning · River · Cloud Run",
    "RoadRisk AI",
    "Plataforma de aprendizaje incremental para estimar la severidad de accidentes vehiculares con datos históricos en Google Cloud Storage."
)

status_tone = "ok" if st.session_state.model_status_type == "good" else "warn"
render_status_banner(st.session_state.model_status, tone=status_tone)

st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin: 0.9rem 0 1rem;">
        <div class="surface" style="margin-bottom: 0; padding: 0.9rem 1rem;">
            <div class="info-label">Bucket fijo</div>
            <div class="info-value">{esc(st.session_state.bucket_name)}</div>
        </div>
        <div class="surface" style="margin-bottom: 0; padding: 0.9rem 1rem;">
            <div class="info-label">Modelo</div>
            <div class="info-value">{esc(MODEL_PATH)}</div>
        </div>
        <div class="surface" style="margin-bottom: 0; padding: 0.9rem 1rem;">
            <div class="info-label">Historial</div>
            <div class="info-value">{esc(HISTORY_PATH)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

with st.sidebar:
    st.markdown("### Configuración")
    st.caption("Parámetros operativos de la sesión actual.")

    st.session_state.project_id = st.text_input(
        "Project ID",
        value=st.session_state.project_id
    )

    st.session_state.bucket_name = st.text_input(
        "Bucket",
        value=st.session_state.bucket_name
    )

    st.session_state.prefix = st.text_input(
        "Prefijo de datos",
        value=st.session_state.prefix
    )

    st.session_state.max_rows = st.number_input(
        "Filas máximas por archivo",
        min_value=100,
        max_value=100000,
        value=int(st.session_state.max_rows),
        step=1000
    )

    st.session_state.balance_training = st.checkbox(
        "Balancear clase mayoritaria",
        value=bool(st.session_state.balance_training)
    )

    st.divider()

    st.caption("Bucket fijo")
    render_path_block(st.session_state.bucket_name)

    st.caption("Modelo")
    render_path_block(f"gs://{st.session_state.bucket_name}/{MODEL_PATH}")

    st.caption("Historial")
    render_path_block(f"gs://{st.session_state.bucket_name}/{HISTORY_PATH}")


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Navegación",
    [
        "Inicio",
        "Predicción",
        "Aprendizaje manual",
        "Entrenamiento incremental",
        "Monitoreo",
        "Configuración técnica"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PAGE: INICIO
# =========================================================

if page == "Inicio":
    if st.session_state.model_status_type == "good":
        st.markdown(
            f'<div class="status-good">{st.session_state.model_status}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-warn">{st.session_state.model_status}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    latest_f1 = None
    latest_recall = None
    latest_file = "Sin archivos procesados"

    if not history.empty:
        if "f1" in history.columns:
            latest_f1 = history["f1"].iloc[-1]
        if "recall" in history.columns:
            latest_recall = history["recall"].iloc[-1]
        if "file" in history.columns:
            latest_file = history["file"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Registros aprendidos",
        f"{model_bundle.get('rows_trained', 0):,}"
    )

    c2.metric(
        "Archivos procesados",
        f"{len(history):,}" if not history.empty else "0"
    )

    c3.metric(
        "Recall más reciente",
        "N/D" if latest_recall is None else f"{latest_recall:.3f}"
    )

    c4.metric(
        "F1 más reciente",
        "N/D" if latest_f1 is None else f"{latest_f1:.3f}"
    )

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Evolución del modelo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Métricas calculadas después de cada partición temporal procesada.</div>',
            unsafe_allow_html=True
        )

        if not history.empty:
            chart_cols = [
                col for col in ["accuracy", "precision", "recall", "f1"]
                if col in history.columns
            ]

            if chart_cols:
                st.line_chart(history[chart_cols])
            else:
                st.info("No hay métricas disponibles todavía.")
        else:
            st.info("Procesa un archivo para empezar a construir el historial.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Estado operativo</div>', unsafe_allow_html=True)
        st.write("Último archivo:", latest_file)
        st.write("Modelo:", model_bundle.get("model_name"))
        st.write("Creado:", model_bundle.get("created_at"))
        st.write("Último entrenamiento:", model_bundle.get("last_training_at"))
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: PREDICCION
# =========================================================

elif page == "Predicción":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Nueva predicción de severidad</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Captura las condiciones disponibles del accidente.</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):
        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown("#### Condiciones físicas")
            distance = st.number_input("Distancia afectada (mi)", value=0.5)
            temperature = st.number_input("Temperatura (F)", value=70.0)
            humidity = st.number_input("Humedad (%)", value=60.0)
            visibility = st.number_input("Visibilidad (mi)", value=10.0)
            wind_speed = st.number_input("Velocidad del viento (mph)", value=5.0)

        with g2:
            st.markdown("#### Ubicación y clima")
            state = st.text_input("Estado", value="CA")
            weather = st.text_input("Condición climática", value="Clear")
            year_month = st.text_input("Mes del evento", value="2023-03")
            hour = st.number_input("Hora", min_value=0, max_value=23, value=18)
            month = st.number_input("Mes", min_value=1, max_value=12, value=3)

        with g3:
            st.markdown("#### Infraestructura")
            dayofweek = st.number_input("Día de la semana", min_value=0, max_value=6, value=2)
            is_weekend = st.selectbox("Fin de semana", [0, 1])
            amenity = st.selectbox("Amenity", [0, 1])
            crossing = st.selectbox("Crossing", [0, 1])
            junction = st.selectbox("Junction", [0, 1], index=1)
            traffic_signal = st.selectbox("Traffic Signal", [0, 1])

        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        x = build_manual_x({
            "distance": distance,
            "temperature": temperature,
            "humidity": humidity,
            "visibility": visibility,
            "wind_speed": wind_speed,
            "state": state,
            "weather": weather,
            "year_month": year_month,
            "amenity": amenity,
            "crossing": crossing,
            "junction": junction,
            "traffic_signal": traffic_signal,
            "hour": hour,
            "month": month,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
        })

        y_pred, proba = predict_one(model_bundle, x)

        st.session_state.last_x = x
        st.session_state.last_prediction = {
            "y_pred": y_pred,
            "proba": dict(proba),
            "created_at": datetime.utcnow().isoformat()
        }

        r1, r2 = st.columns([1, 1])

        with r1:
            if y_pred == 1:
                st.markdown(
                    '<div class="risk-high">RIESGO ALTO<br>Accidente severo probable</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="risk-low">RIESGO BAJO<br>Accidente no severo probable</div>',
                    unsafe_allow_html=True
                )

        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Detalle de predicción</div>', unsafe_allow_html=True)
            st.write("Clase predicha:", y_pred)
            st.write("Probabilidades:", dict(proba))
            st.write("Fecha de predicción:", st.session_state.last_prediction["created_at"])
            st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: APRENDIZAJE MANUAL
# =========================================================

elif page == "Aprendizaje manual":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Aprendizaje manual con etiqueta real</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Permite actualizar el modelo usando el último caso consultado.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.last_x is None:
        st.info("Primero realiza una predicción en la sección Predicción.")
    else:
        st.write("Último caso consultado:")
        st.json(st.session_state.last_x)

        y_true = st.selectbox(
            "Etiqueta real observada",
            [0, 1],
            format_func=lambda x: "No severo" if x == 0 else "Severo"
        )

        if st.button("Actualizar modelo con este caso", type="primary"):
            prev_pred, prev_proba = predict_one(model_bundle, st.session_state.last_x)

            learn_one(model_bundle, st.session_state.last_x, int(y_true))
            save_pickle_to_gcs(model_bundle, st.session_state.bucket_name, MODEL_PATH)

            st.session_state.model_bundle = model_bundle

            st.success("Modelo actualizado y guardado en GCS.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicción previa", prev_pred)
            c2.metric("Etiqueta real", int(y_true))
            c3.metric("Registros aprendidos", f"{model_bundle.get('rows_trained', 0):,}")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: ENTRENAMIENTO INCREMENTAL
# =========================================================

elif page == "Entrenamiento incremental":
    render_surface(
        "Entrenamiento incremental por archivo",
        "Selecciona un archivo CSV concreto para entrenarlo. Los ya procesados se deshabilitan usando el historial persistido."
    )

    st.markdown('<div class="section-title">Operación</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Puedes actualizar el listado, procesar archivos individuales o borrar el modelo e historial para empezar de cero.</div>',
        unsafe_allow_html=True
    )

    op_a, op_b, op_c = st.columns([1, 1.2, 1])

    with op_a:
        if st.button("Actualizar listado", type="secondary"):
            with st.spinner("Consultando Google Cloud Storage..."):
                st.session_state.files = list_csv_files(
                    st.session_state.bucket_name,
                    st.session_state.prefix
                )
            st.success(f"Archivos encontrados: {len(st.session_state.files)}")

    with op_b:
        st.caption("Origen")
        render_path_block(f"gs://{st.session_state.bucket_name}/{st.session_state.prefix}")

    with op_c:
        reset_all = st.button("Borrar modelo e historial", type="secondary")

    if reset_all:
        delete_blob_if_exists(st.session_state.bucket_name, MODEL_PATH)
        delete_blob_if_exists(st.session_state.bucket_name, HISTORY_PATH)

        st.session_state.model_bundle = new_model_bundle()
        st.session_state.history = pd.DataFrame()
        st.session_state.files = []
        st.session_state.last_x = None
        st.session_state.last_prediction = None

        st.success("Modelo e historial borrados.")
        st.rerun()

    if "files" not in st.session_state or not st.session_state.files:
        with st.spinner("Listando archivos automáticamente..."):
            st.session_state.files = list_csv_files(
                st.session_state.bucket_name,
                st.session_state.prefix
            )

    processed_files = set()
    if not st.session_state.history.empty and "blob_name" in st.session_state.history.columns:
        processed_files = set(st.session_state.history["blob_name"].dropna().astype(str))

    total_files = len(st.session_state.files)
    processed_count = sum(1 for blob in st.session_state.files if blob in processed_files)
    pending_count = total_files - processed_count

    st.markdown('<div class="section-title">Resumen</div>', unsafe_allow_html=True)

    top_a, top_b, top_c = st.columns(3)
    top_a.metric("Archivos listados", f"{total_files:,}")
    top_b.metric("Procesados", f"{processed_count:,}")
    top_c.metric("Pendientes", f"{pending_count:,}")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-title">Archivos disponibles</div>', unsafe_allow_html=True)
        if not st.session_state.files:
            st.info("No se encontraron archivos CSV con el prefijo configurado.")
        else:
            for idx, blob_name in enumerate(st.session_state.files):
                filename = blob_name.split("/")[-1]
                is_processed = blob_name in processed_files

                st.markdown('<div class="surface">', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="surface-head">
                        <div class="surface-title">{esc(filename)}</div>
                        <div class="surface-subtitle">{esc(blob_name)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if is_processed:
                    render_status_banner("Ya procesado", tone="ok")
                else:
                    render_status_banner("Pendiente de procesar", tone="info")

                if st.button(
                    "Procesar archivo",
                    key=f"process_selected_{idx}_{blob_name}",
                    type="primary",
                    disabled=is_processed
                ):
                    with st.spinner("Entrenando modelo incremental..."):
                        model_bundle, result = train_on_file(
                            model_bundle=model_bundle,
                            bucket_name=st.session_state.bucket_name,
                            blob_name=blob_name,
                            max_rows=int(st.session_state.max_rows),
                            balance_training=bool(st.session_state.balance_training)
                        )

                    save_pickle_to_gcs(
                        model_bundle,
                        st.session_state.bucket_name,
                        MODEL_PATH
                    )

                    hist = st.session_state.history.copy()
                    hist = pd.concat([hist, pd.DataFrame([result])], ignore_index=True)
                    st.session_state.history = hist

                    save_history_to_gcs(
                        hist,
                        st.session_state.bucket_name,
                        HISTORY_PATH
                    )

                    st.session_state.model_bundle = model_bundle
                    st.success(f"Archivo procesado correctamente: {filename}")
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="surface-head"><div class="surface-title">Configuración de proceso</div><div class="surface-subtitle">Fuente operativa y criterios actuales.</div></div>', unsafe_allow_html=True)
        st.caption("Origen")
        render_path_block(f"gs://{st.session_state.bucket_name}/{st.session_state.prefix}")
        st.caption("Modelo")
        render_path_block(f"gs://{st.session_state.bucket_name}/{MODEL_PATH}")
        st.caption("Historial")
        render_path_block(f"gs://{st.session_state.bucket_name}/{HISTORY_PATH}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Entrenamiento incremental por archivo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Selecciona cualquier CSV disponible. Los archivos ya procesados quedan deshabilitados y se toman del historial guardado.</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        if st.button("Actualizar listado", type="secondary"):
            with st.spinner("Consultando Google Cloud Storage..."):
                st.session_state.files = list_csv_files(
                    st.session_state.bucket_name,
                    st.session_state.prefix
                )
            st.success(f"Archivos encontrados: {len(st.session_state.files)}")

    with b:
        st.caption("Fuente de verdad")
        st.code(f"gs://{st.session_state.bucket_name}/{st.session_state.prefix}")

    with c:
        reset_all = st.button("Reiniciar modelo e historial")

    if reset_all:
        delete_blob_if_exists(st.session_state.bucket_name, MODEL_PATH)
        delete_blob_if_exists(st.session_state.bucket_name, HISTORY_PATH)

        st.session_state.model_bundle = new_model_bundle()
        st.session_state.history = pd.DataFrame()
        st.session_state.files = []
        st.session_state.last_x = None
        st.session_state.last_prediction = None

        st.success("Modelo e historial reiniciados.")
        st.rerun()

    if not st.session_state.files:
        with st.spinner("Listando archivos automáticamente..."):
            st.session_state.files = list_csv_files(
                st.session_state.bucket_name,
                st.session_state.prefix
            )

    processed_files = set()
    if not st.session_state.history.empty and "blob_name" in st.session_state.history.columns:
        processed_files = set(st.session_state.history["blob_name"].dropna().astype(str))

    total_files = len(st.session_state.files)
    processed_count = sum(1 for blob in st.session_state.files if blob in processed_files)
    pending_count = total_files - processed_count

    s1, s2, s3 = st.columns(3)
    s1.metric("Archivos listados", f"{total_files:,}")
    s2.metric("Procesados", f"{processed_count:,}")
    s3.metric("Pendientes", f"{pending_count:,}")

    if not st.session_state.files:
        st.info("No se encontraron archivos CSV con el prefijo configurado.")
    else:
        st.markdown("#### Archivos disponibles")
        cols = st.columns(2)

        for idx, blob_name in enumerate(st.session_state.files):
            is_processed = blob_name in processed_files
            col = cols[idx % 2]

            with col:
                st.markdown('<div class="surface">', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="surface-head">
                        <div class="surface-title">{esc(blob_name.split('/')[-1])}</div>
                        <div class="surface-subtitle">{esc(blob_name)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if is_processed:
                    render_status_banner("Ya procesado", tone="ok")
                else:
                    render_status_banner("Pendiente de procesar", tone="info")

                if st.button(
                    "Procesar archivo",
                    key=f"process_{idx}_{blob_name}",
                    type="primary",
                    disabled=is_processed
                ):
                    with st.spinner("Entrenando modelo incremental..."):
                        model_bundle, result = train_on_file(
                            model_bundle=model_bundle,
                            bucket_name=st.session_state.bucket_name,
                            blob_name=blob_name,
                            max_rows=int(st.session_state.max_rows),
                            balance_training=bool(st.session_state.balance_training)
                        )

                    save_pickle_to_gcs(
                        model_bundle,
                        st.session_state.bucket_name,
                        MODEL_PATH
                    )

                    hist = st.session_state.history.copy()
                    hist = pd.concat([hist, pd.DataFrame([result])], ignore_index=True)
                    st.session_state.history = hist

                    save_history_to_gcs(
                        hist,
                        st.session_state.bucket_name,
                        HISTORY_PATH
                    )

                    st.session_state.model_bundle = model_bundle

                    st.success(f"Archivo procesado correctamente: {blob_name}")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{result['accuracy']:.3f}")
                    m2.metric("Recall", f"{result['recall']:.3f}")
                    m3.metric("F1", f"{result['f1']:.3f}")
                    m4.metric("Filas aprendidas", f"{result['rows_learned']:,}")
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: MONITOREO
# =========================================================

elif page == "Monitoreo":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Monitoreo del desempeño</div>', unsafe_allow_html=True)

    if history.empty:
        st.info("No hay historial. Procesa al menos un archivo.")
    else:
        metric_cols = [
            col for col in ["accuracy", "precision", "recall", "f1"]
            if col in history.columns
        ]

        if metric_cols:
            st.line_chart(history[metric_cols])

        if "severe_rate_file" in history.columns:
            st.markdown("#### Tasa de accidentes severos por archivo")
            st.line_chart(history[["severe_rate_file"]])

        st.markdown("#### Historial completo")
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Descargar historial CSV",
            data=csv_data,
            file_name="history_incremental.csv",
            mime="text/csv"
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE: CONFIGURACION TECNICA
# =========================================================

elif page == "Configuración técnica":
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Infraestructura</div>', unsafe_allow_html=True)
        st.write("Project ID:", st.session_state.project_id)
        st.write("Bucket:", st.session_state.bucket_name)
        st.write("Prefijo:", st.session_state.prefix)
        st.write("Modelo:", model_bundle.get("model_name"))
        st.write("Registros aprendidos:", f"{model_bundle.get('rows_trained', 0):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Artefactos</div>', unsafe_allow_html=True)
        st.markdown("Modelo")
        st.markdown(
            f'<div class="small-code">gs://{st.session_state.bucket_name}/{MODEL_PATH}</div>',
            unsafe_allow_html=True
        )
        st.markdown("Historial")
        st.markdown(
            f'<div class="small-code">gs://{st.session_state.bucket_name}/{HISTORY_PATH}</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Variables del modelo</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "feature": SELECTED_FEATURES,
            "grupo": (
                ["numérica"] * len(NUM_FEATURES)
                + ["categórica"] * len(CAT_FEATURES)
                + ["booleana"] * len(BOOL_FEATURES)
                + ["temporal"] * len(DERIVED_TIME_FEATURES)
            )
        }),
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="app-footer">
        <span>RoadRisk AI · River · Google Cloud Run · Google Cloud Storage</span>
    </div>
    """,
    unsafe_allow_html=True
)
