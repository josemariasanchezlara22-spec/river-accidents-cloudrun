import io
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
        --card: #ffffff;
        --primary: #0f2742;
        --primary-2: #1f6feb;
        --muted: #6b7280;
        --border: #e5e7eb;
        --green: #0f8a5f;
        --green-bg: #e9f8f1;
        --red: #b42318;
        --red-bg: #fdecec;
        --orange: #b54708;
        --orange-bg: #fff4e5;
        --blue-bg: #eef5ff;
    }

    .stApp {
        background: var(--bg);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    .topbar {
        background: linear-gradient(135deg, #0f2742 0%, #173b63 55%, #1f6feb 100%);
        color: white;
        padding: 2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(15, 39, 66, 0.20);
    }

    .topbar h1 {
        color: white;
        font-size: 2.4rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .topbar p {
        color: #dbeafe;
        margin-top: 0.5rem;
        font-size: 1.05rem;
        max-width: 900px;
    }

    .pill {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        color: white;
        border: 1px solid rgba(255,255,255,0.25);
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 39, 66, 0.06);
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 750;
        color: var(--primary);
        margin-bottom: 0.35rem;
    }

    .card-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .risk-high {
        background: var(--red-bg);
        border: 1px solid #f5b5b0;
        color: var(--red);
        border-radius: 18px;
        padding: 1.3rem;
        font-weight: 800;
        font-size: 1.25rem;
        text-align: center;
    }

    .risk-low {
        background: var(--green-bg);
        border: 1px solid #a8e6c8;
        color: var(--green);
        border-radius: 18px;
        padding: 1.3rem;
        font-weight: 800;
        font-size: 1.25rem;
        text-align: center;
    }

    .status-good {
        background: var(--green-bg);
        color: var(--green);
        padding: 0.7rem 0.9rem;
        border-radius: 14px;
        font-weight: 700;
    }

    .status-neutral {
        background: var(--blue-bg);
        color: var(--primary-2);
        padding: 0.7rem 0.9rem;
        border-radius: 14px;
        font-weight: 700;
    }

    .status-warn {
        background: var(--orange-bg);
        color: var(--orange);
        padding: 0.7rem 0.9rem;
        border-radius: 14px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(15, 39, 66, 0.05);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        color: #0f2742;
        font-weight: 800;
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280;
    }

    .small-code {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 0.8rem;
        border-radius: 12px;
        font-family: monospace;
        font-size: 0.86rem;
        color: #334155;
    }

    .nav-note {
        color: #6b7280;
        font-size: 0.9rem;
    }

    button[kind="primary"] {
        border-radius: 999px !important;
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

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

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

st.markdown(
    """
    <div class="topbar">
        <div class="pill">Online Learning · River · Cloud Run</div>
        <h1>RoadRisk AI</h1>
        <p>
            Plataforma de aprendizaje incremental para estimar la severidad de accidentes
            vehiculares usando datos históricos de US Accidents y almacenamiento en Google Cloud Storage.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

with st.sidebar:
    st.title("Configuración")

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

    st.caption("Modelo")
    st.code(f"gs://{st.session_state.bucket_name}/{MODEL_PATH}")

    st.caption("Historial")
    st.code(f"gs://{st.session_state.bucket_name}/{HISTORY_PATH}")


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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Entrenamiento incremental por archivo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Cada archivo CSV mensual se procesa como una nueva partición temporal.</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        if st.button("Listar archivos", type="secondary"):
            with st.spinner("Consultando Google Cloud Storage..."):
                st.session_state.files = list_csv_files(
                    st.session_state.bucket_name,
                    st.session_state.prefix
                )
                st.session_state.file_index = 0

            st.success(f"Archivos encontrados: {len(st.session_state.files)}")

    with b:
        process_next = st.button("Procesar siguiente archivo", type="primary")

    with c:
        reset_all = st.button("Reiniciar modelo e historial")

    if reset_all:
        delete_blob_if_exists(st.session_state.bucket_name, MODEL_PATH)
        delete_blob_if_exists(st.session_state.bucket_name, HISTORY_PATH)

        st.session_state.model_bundle = new_model_bundle()
        st.session_state.history = pd.DataFrame()
        st.session_state.files = []
        st.session_state.file_index = 0
        st.session_state.last_x = None
        st.session_state.last_prediction = None

        st.success("Modelo e historial reiniciados.")
        st.rerun()

    if process_next:
        if not st.session_state.files:
            with st.spinner("Listando archivos automáticamente..."):
                st.session_state.files = list_csv_files(
                    st.session_state.bucket_name,
                    st.session_state.prefix
                )
                st.session_state.file_index = 0

        files = st.session_state.files
        idx = st.session_state.file_index

        if idx >= len(files):
            st.success("Todos los archivos ya fueron procesados.")
        else:
            blob_name = files[idx]

            st.info(f"Procesando {idx + 1}/{len(files)}: {blob_name}")

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

            st.session_state.file_index += 1
            st.session_state.model_bundle = model_bundle

            st.success("Archivo procesado correctamente.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{result['accuracy']:.3f}")
            m2.metric("Recall", f"{result['recall']:.3f}")
            m3.metric("F1", f"{result['f1']:.3f}")
            m4.metric("Filas aprendidas", f"{result['rows_learned']:,}")

    st.markdown("</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Archivo actual", st.session_state.file_index)
    s2.metric("Archivos cargados", len(st.session_state.files))
    s3.metric("Filas máximas por archivo", f"{int(st.session_state.max_rows):,}")

    if st.session_state.files and st.session_state.file_index < len(st.session_state.files):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Siguiente partición</div>', unsafe_allow_html=True)
        st.write(st.session_state.files[st.session_state.file_index])
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


st.caption("RoadRisk AI · River · Google Cloud Run · Google Cloud Storage")