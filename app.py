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
# CONFIGURACION VISUAL
# =========================================================

st.set_page_config(
    page_title="US Accidents | Modelo Incremental",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .app-header {
        background: linear-gradient(90deg, #072146 0%, #1464A5 100%);
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .app-header h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }

    .app-header p {
        color: #eaf3ff;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e6eaf0;
        box-shadow: 0 2px 8px rgba(7, 33, 70, 0.05);
        margin-bottom: 1rem;
    }

    .status-ok {
        background-color: #e8f6ef;
        color: #1b7f4c;
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        font-weight: 600;
    }

    .status-warn {
        background-color: #fff4de;
        color: #9a6100;
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        font-weight: 600;
    }

    .status-risk {
        background-color: #fdeaea;
        color: #b42318;
        padding: 0.6rem 0.8rem;
        border-radius: 10px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-header">
        <h1>US Accidents | Aprendizaje Incremental con River</h1>
        <p>
            Aplicación desplegable en Cloud Run para predicción online, entrenamiento incremental
            y persistencia del modelo en Google Cloud Storage.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CONSTANTES
# =========================================================

DEFAULT_PROJECT_ID = os.getenv(
    "PROJECT_ID",
    "ml-big-data-q2-up"
)

DEFAULT_BUCKET_NAME = os.getenv(
    "BUCKET_NAME",
    "us-accidents-am-up-02"
)

DEFAULT_PREFIX = os.getenv(
    "DATA_PREFIX",
    "raw/"
)

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
    client = get_storage_client()
    return client.bucket(bucket_name)


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

    files = [
        blob.name
        for blob in blobs
        if blob.name.lower().endswith(".csv")
    ]

    return sorted(files)


def read_csv_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data), low_memory=False)


# =========================================================
# MODELO RIVER
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
# PREPARACION DE DATOS
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

    keep_cols = SELECTED_FEATURES + [TARGET_BINARY, TARGET_ORIGINAL]

    return df[keep_cols]


def row_to_x(row) -> dict:
    x = {}

    for feature in SELECTED_FEATURES:
        value = row[feature]

        if pd.isna(value):
            if feature in CAT_FEATURES:
                value = "Unknown"
            else:
                value = 0

        if feature in CAT_FEATURES:
            x[feature] = str(value)
        else:
            x[feature] = float(value)

    return x


def build_manual_x(
    distance,
    temperature,
    humidity,
    visibility,
    wind_speed,
    state,
    weather,
    year_month,
    amenity,
    crossing,
    junction,
    traffic_signal,
    hour,
    month,
    dayofweek,
    is_weekend
):
    return {
        "Distance(mi)": float(distance),
        "Temperature(F)": float(temperature),
        "Humidity(%)": float(humidity),
        "Visibility(mi)": float(visibility),
        "Wind_Speed(mph)": float(wind_speed),
        "State": str(state),
        "Weather_Condition": str(weather),
        "year_month": str(year_month),
        "Amenity": float(amenity),
        "Crossing": float(crossing),
        "Junction": float(junction),
        "Traffic_Signal": float(traffic_signal),
        "hour": float(hour),
        "month": float(month),
        "dayofweek": float(dayofweek),
        "is_weekend": float(is_weekend),
    }


# =========================================================
# ENTRENAMIENTO POR ARCHIVO
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
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Configuracion")

    project_id = st.text_input(
        "Proyecto GCP",
        value=DEFAULT_PROJECT_ID
    )

    bucket_name = st.text_input(
        "Bucket GCS",
        value=DEFAULT_BUCKET_NAME
    )

    prefix = st.text_input(
        "Prefijo de datos",
        value=DEFAULT_PREFIX
    )

    max_rows = st.number_input(
        "Filas maximas por archivo",
        min_value=100,
        max_value=100000,
        value=8000,
        step=1000
    )

    balance_training = st.checkbox(
        "Balancear clase mayoritaria",
        value=True
    )

    st.divider()

    st.caption("Artefactos")
    st.code(f"gs://{bucket_name}/{MODEL_PATH}")
    st.code(f"gs://{bucket_name}/{HISTORY_PATH}")


# =========================================================
# INICIALIZACION DE SESION
# =========================================================

if "model_bundle" not in st.session_state:
    try:
        loaded_model = load_pickle_from_gcs(bucket_name, MODEL_PATH)

        if loaded_model is None:
            loaded_model = new_model_bundle()
            st.session_state.model_load_status = "new"
        else:
            st.session_state.model_load_status = "loaded"

        st.session_state.model_bundle = loaded_model

    except Exception as e:
        st.session_state.model_bundle = new_model_bundle()
        st.session_state.model_load_status = "error"
        st.session_state.model_load_error = str(e)

if "history" not in st.session_state:
    try:
        st.session_state.history = load_history_from_gcs(bucket_name, HISTORY_PATH)
    except Exception:
        st.session_state.history = pd.DataFrame()

if "files" not in st.session_state:
    st.session_state.files = []

if "file_index" not in st.session_state:
    st.session_state.file_index = 0

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_x" not in st.session_state:
    st.session_state.last_x = None

model_bundle = st.session_state.model_bundle
history = st.session_state.history


# =========================================================
# COMPONENTES VISUALES
# =========================================================

def render_model_status_card():
    status = st.session_state.get("model_load_status", "new")

    if status == "loaded":
        st.markdown(
            '<div class="status-ok">Modelo cargado correctamente desde GCS</div>',
            unsafe_allow_html=True
        )
    elif status == "new":
        st.markdown(
            '<div class="status-warn">No habia modelo previo. Se inicio un modelo nuevo.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-risk">No se pudo cargar el modelo desde GCS. Se inicio modelo local.</div>',
            unsafe_allow_html=True
        )
        st.caption(st.session_state.get("model_load_error", ""))


def render_prediction_inputs(prefix_key: str = "pred"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Variables numericas**")
        distance = st.number_input("Distance(mi)", value=0.5, key=f"{prefix_key}_distance")
        temperature = st.number_input("Temperature(F)", value=70.0, key=f"{prefix_key}_temp")
        humidity = st.number_input("Humidity(%)", value=60.0, key=f"{prefix_key}_humidity")
        visibility = st.number_input("Visibility(mi)", value=10.0, key=f"{prefix_key}_visibility")
        wind_speed = st.number_input("Wind_Speed(mph)", value=5.0, key=f"{prefix_key}_wind")

    with col2:
        st.markdown("**Variables categoricas**")
        state = st.text_input("State", value="CA", key=f"{prefix_key}_state")
        weather = st.text_input("Weather_Condition", value="Clear", key=f"{prefix_key}_weather")
        year_month = st.text_input("year_month", value="2023-03", key=f"{prefix_key}_year_month")

        hour = st.number_input(
            "hour",
            min_value=0,
            max_value=23,
            value=18,
            key=f"{prefix_key}_hour"
        )

        month = st.number_input(
            "month",
            min_value=1,
            max_value=12,
            value=3,
            key=f"{prefix_key}_month"
        )

    with col3:
        st.markdown("**Infraestructura y tiempo**")

        dayofweek = st.number_input(
            "dayofweek",
            min_value=0,
            max_value=6,
            value=2,
            key=f"{prefix_key}_dow"
        )

        is_weekend = st.selectbox("is_weekend", [0, 1], index=0, key=f"{prefix_key}_weekend")
        amenity = st.selectbox("Amenity", [0, 1], index=0, key=f"{prefix_key}_amenity")
        crossing = st.selectbox("Crossing", [0, 1], index=0, key=f"{prefix_key}_crossing")
        junction = st.selectbox("Junction", [0, 1], index=1, key=f"{prefix_key}_junction")
        traffic_signal = st.selectbox(
            "Traffic_Signal",
            [0, 1],
            index=0,
            key=f"{prefix_key}_signal"
        )

    x = build_manual_x(
        distance=distance,
        temperature=temperature,
        humidity=humidity,
        visibility=visibility,
        wind_speed=wind_speed,
        state=state,
        weather=weather,
        year_month=year_month,
        amenity=amenity,
        crossing=crossing,
        junction=junction,
        traffic_signal=traffic_signal,
        hour=hour,
        month=month,
        dayofweek=dayofweek,
        is_weekend=is_weekend
    )

    return x


# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "Dashboard",
        "Prediccion online",
        "Aprendizaje manual",
        "Entrenamiento por archivo",
        "Historial y artefactos",
        "Informacion del proyecto"
    ]
)


# =========================================================
# TAB 1 - DASHBOARD
# =========================================================

with tabs[0]:
    st.subheader("Dashboard del modelo")

    render_model_status_card()

    st.markdown("")

    total_files_processed = 0
    last_f1 = None
    last_recall = None
    last_accuracy = None

    if not history.empty:
        total_files_processed = len(history)

        if "f1" in history.columns:
            last_f1 = history["f1"].iloc[-1]

        if "recall" in history.columns:
            last_recall = history["recall"].iloc[-1]

        if "accuracy" in history.columns:
            last_accuracy = history["accuracy"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registros aprendidos",
        f"{model_bundle.get('rows_trained', 0):,}"
    )

    col2.metric(
        "Archivos procesados",
        f"{total_files_processed:,}"
    )

    col3.metric(
        "Recall ultimo archivo",
        "N/D" if last_recall is None else f"{last_recall:.3f}"
    )

    col4.metric(
        "F1 ultimo archivo",
        "N/D" if last_f1 is None else f"{last_f1:.3f}"
    )

    st.markdown("")

    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Evolucion de metricas")

        if not history.empty:
            chart_cols = [
                col
                for col in ["accuracy", "precision", "recall", "f1"]
                if col in history.columns
            ]

            if chart_cols:
                st.line_chart(history[chart_cols])
            else:
                st.info("Aun no hay columnas de metricas para graficar.")
        else:
            st.info("Aun no existe historial. Procesa al menos un archivo.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Estado operativo")

        st.write("Modelo:", model_bundle.get("model_name"))
        st.write("Creado:", model_bundle.get("created_at"))
        st.write("Ultimo entrenamiento:", model_bundle.get("last_training_at"))
        st.write("Bucket:", bucket_name)
        st.write("Prefijo:", prefix)

        st.markdown("</div>", unsafe_allow_html=True)

    if not history.empty:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Ultimos archivos procesados")
        st.dataframe(history.tail(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2 - PREDICCION ONLINE
# =========================================================

with tabs[1]:
    st.subheader("Prediccion online")

    st.markdown(
        """
        Captura las variables disponibles al momento del accidente.
        El modelo devuelve si el accidente seria clasificado como severo o no severo.
        """
    )

    x_pred = render_prediction_inputs(prefix_key="predict")

    col_button, col_result = st.columns([1, 2])

    with col_button:
        predict_clicked = st.button("Ejecutar prediccion", type="primary")

    if predict_clicked:
        y_pred, proba = predict_one(model_bundle, x_pred)

        st.session_state.last_prediction = {
            "x": x_pred,
            "y_pred": y_pred,
            "proba": proba,
            "predicted_at": datetime.utcnow().isoformat()
        }

        st.session_state.last_x = x_pred

    if st.session_state.last_prediction is not None:
        pred = st.session_state.last_prediction["y_pred"]
        proba = st.session_state.last_prediction["proba"]

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(
                '<div class="status-risk">Prediccion: accidente severo</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-ok">Prediccion: accidente no severo</div>',
                unsafe_allow_html=True
            )

        st.write("Clase predicha:", pred)
        st.write("Probabilidades estimadas:", dict(proba))

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3 - APRENDIZAJE MANUAL
# =========================================================

with tabs[2]:
    st.subheader("Aprendizaje manual con etiqueta real")

    st.markdown(
        """
        Esta seccion simula el caso en que un usuario consulta una prediccion y,
        posteriormente, se conoce la etiqueta real. El modelo aprende un registro nuevo
        sin reentrenar desde cero.
        """
    )

    if st.session_state.last_x is None:
        st.info("Primero ejecuta una prediccion en la pestana 'Prediccion online'.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Ultimo caso consultado")
        st.json(st.session_state.last_x)
        st.markdown("</div>", unsafe_allow_html=True)

        y_true_manual = st.selectbox(
            "Etiqueta real observada",
            [0, 1],
            format_func=lambda value: "No severo" if value == 0 else "Severo"
        )

        if st.button("Actualizar modelo con este caso", type="primary"):
            x_manual = st.session_state.last_x

            previous_prediction, previous_proba = predict_one(model_bundle, x_manual)

            learn_one(model_bundle, x_manual, int(y_true_manual))
            save_pickle_to_gcs(model_bundle, bucket_name, MODEL_PATH)

            st.session_state.model_bundle = model_bundle

            st.success("Modelo actualizado y guardado en GCS.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediccion previa", previous_prediction)
            col2.metric("Etiqueta real", int(y_true_manual))
            col3.metric("Registros aprendidos", f"{model_bundle.get('rows_trained', 0):,}")

            st.write("Probabilidades previas:", dict(previous_proba))


# =========================================================
# TAB 4 - ENTRENAMIENTO POR ARCHIVO
# =========================================================

with tabs[3]:
    st.subheader("Entrenamiento incremental por archivo mensual")

    st.markdown(
        """
        Procesa archivos CSV desde Google Cloud Storage. Cada archivo representa una particion temporal.
        El flujo aplicado es: **predecir → evaluar → aprender**.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Listar archivos CSV"):
            with st.spinner("Buscando archivos en GCS..."):
                files = list_csv_files(bucket_name, prefix)

            st.session_state.files = files
            st.session_state.file_index = 0

            st.success(f"Archivos encontrados: {len(files)}")

    with col2:
        process_clicked = st.button("Procesar siguiente archivo", type="primary")

    with col3:
        reset_clicked = st.button("Reiniciar modelo e historial")

    if reset_clicked:
        delete_blob_if_exists(bucket_name, MODEL_PATH)
        delete_blob_if_exists(bucket_name, HISTORY_PATH)

        st.session_state.model_bundle = new_model_bundle()
        st.session_state.history = pd.DataFrame()
        st.session_state.files = []
        st.session_state.file_index = 0
        st.session_state.last_prediction = None
        st.session_state.last_x = None

        st.success("Modelo e historial reiniciados correctamente.")
        st.rerun()

    if process_clicked:
        if not st.session_state.files:
            with st.spinner("Listando archivos automaticamente..."):
                st.session_state.files = list_csv_files(bucket_name, prefix)
                st.session_state.file_index = 0

        files = st.session_state.files
        idx = st.session_state.file_index

        if idx >= len(files):
            st.success("Todos los archivos ya fueron procesados.")
        else:
            blob_name = files[idx]

            st.info(f"Procesando archivo {idx + 1}/{len(files)}: {blob_name}")

            with st.spinner("Entrenando incrementalmente..."):
                model_bundle, result = train_on_file(
                    model_bundle=model_bundle,
                    bucket_name=bucket_name,
                    blob_name=blob_name,
                    max_rows=int(max_rows),
                    balance_training=balance_training
                )

            save_pickle_to_gcs(model_bundle, bucket_name, MODEL_PATH)

            hist = st.session_state.history.copy()
            result_df = pd.DataFrame([result])
            hist = pd.concat([hist, result_df], ignore_index=True)

            st.session_state.history = hist
            save_history_to_gcs(hist, bucket_name, HISTORY_PATH)

            st.session_state.file_index += 1
            st.session_state.model_bundle = model_bundle

            st.success("Archivo procesado. Modelo e historial guardados en GCS.")

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy", f"{result['accuracy']:.3f}")
            col_b.metric("Recall", f"{result['recall']:.3f}")
            col_c.metric("F1", f"{result['f1']:.3f}")
            col_d.metric("Filas aprendidas", f"{result['rows_learned']:,}")

    st.markdown("---")

    col_status_1, col_status_2, col_status_3 = st.columns(3)

    col_status_1.metric(
        "Indice de archivo actual",
        st.session_state.file_index
    )

    col_status_2.metric(
        "Archivos cargados",
        len(st.session_state.files)
    )

    col_status_3.metric(
        "Max filas por archivo",
        f"{int(max_rows):,}"
    )

    if st.session_state.files:
        next_idx = st.session_state.file_index

        if next_idx < len(st.session_state.files):
            st.write("Siguiente archivo:", st.session_state.files[next_idx])
        else:
            st.write("Siguiente archivo: no quedan archivos pendientes.")


# =========================================================
# TAB 5 - HISTORIAL Y ARTEFACTOS
# =========================================================

with tabs[4]:
    st.subheader("Historial y artefactos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Modelo")
        st.code(f"gs://{bucket_name}/{MODEL_PATH}")
        st.write("Registros aprendidos:", f"{model_bundle.get('rows_trained', 0):,}")
        st.write("Ultima actualizacion:", model_bundle.get("last_training_at"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Historial")
        st.code(f"gs://{bucket_name}/{HISTORY_PATH}")

        if history.empty:
            st.write("No hay historial guardado.")
        else:
            st.write("Filas de historial:", len(history))

        st.markdown("</div>", unsafe_allow_html=True)

    if not history.empty:
        st.markdown("### Tabla completa de historial")
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar historial CSV",
            data=csv_data,
            file_name="history_incremental.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 6 - INFORMACION
# =========================================================

with tabs[5]:
    st.subheader("Informacion del proyecto")

    st.markdown(
        """
        ### Objetivo

        Desplegar una solucion de aprendizaje incremental usando River sobre Google Cloud Run.
        El modelo clasifica accidentes vehiculares como severos o no severos.

        ### Problema

        Clasificacion binaria:

        - `is_severe = 1` si `Severity >= 3`
        - `is_severe = 0` si `Severity < 3`

        ### Modelo

        - Libreria: River
        - Algoritmo: Hoeffding Tree Classifier
        - Codificacion categorica: OneHotEncoder incremental
        - Entrenamiento: registro por registro
        - Persistencia: Google Cloud Storage

        ### Flujo de aprendizaje

        1. Se carga un archivo mensual desde GCS.
        2. Se prepara el dataset.
        3. Para cada registro, el modelo predice primero.
        4. Se actualizan metricas.
        5. El modelo aprende con `learn_one`.
        6. Se guarda el modelo actualizado en GCS.
        """
    )

    st.markdown("### Variables usadas")
    st.dataframe(
        pd.DataFrame(
            {
                "feature": SELECTED_FEATURES,
                "tipo": (
                    ["numerica"] * len(NUM_FEATURES)
                    + ["categorica"] * len(CAT_FEATURES)
                    + ["booleana"] * len(BOOL_FEATURES)
                    + ["temporal"] * len(DERIVED_TIME_FEATURES)
                )
            }
        ),
        use_container_width=True
    )


st.caption("Cloud Run · Streamlit · River · Google Cloud Storage")