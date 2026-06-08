import io
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from river import preprocessing, tree, metrics


# =========================================================
# CONFIGURACION GENERAL
# =========================================================

st.set_page_config(
    page_title="US Accidents - Aprendizaje Incremental",
    page_icon="🚗",
    layout="wide"
)

st.title("Aprendizaje Incremental con River en Cloud Run")
st.markdown(
    """
    Aplicación para entrenar e interactuar con un modelo incremental de clasificación binaria
    sobre el dataset **US Accidents 2016-2023**.
    
    Flujo usado: **predicción → evaluación → aprendizaje incremental**.
    """
)


# =========================================================
# CONSTANTES DEL PROYECTO
# =========================================================

DEFAULT_PROJECT_ID = "am-up-01"
DEFAULT_BUCKET_NAME = "us-accidents-am-up-01"
DEFAULT_PREFIX = "raw/"
MODEL_PATH = "models/modelo_incremental_ht.pkl"
HISTORY_PATH = "models/history_incremental.csv"

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

def get_storage_client():
    return storage.Client()


def load_pickle_from_gcs(bucket_name: str, blob_name: str):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    data = blob.download_as_bytes()
    return pickle.loads(data)


def save_pickle_to_gcs(obj, bucket_name: str, blob_name: str):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pickle.dumps(obj))


def load_history_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return pd.DataFrame()

    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def save_history_to_gcs(df: pd.DataFrame, bucket_name: str, blob_name: str):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv"
    )


def delete_blob_if_exists(bucket_name: str, blob_name: str):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if blob.exists():
        blob.delete()


def list_csv_files(bucket_name: str, prefix: str):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))

    files = [
        blob.name
        for blob in blobs
        if blob.name.lower().endswith(".csv")
    ]

    return sorted(files)


def read_csv_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
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
        "rows_trained": 0
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

st.sidebar.header("Configuracion")

project_id = st.sidebar.text_input(
    "Proyecto GCP",
    value=DEFAULT_PROJECT_ID
)

bucket_name = st.sidebar.text_input(
    "Bucket GCS",
    value=DEFAULT_BUCKET_NAME
)

prefix = st.sidebar.text_input(
    "Prefijo de archivos CSV",
    value=DEFAULT_PREFIX
)

max_rows = st.sidebar.number_input(
    "Maximo de filas por archivo",
    min_value=100,
    max_value=100000,
    value=8000,
    step=1000
)

balance_training = st.sidebar.checkbox(
    "Aplicar balanceo por submuestreo de clase 0",
    value=True
)


# =========================================================
# INICIALIZACION
# =========================================================

if "model_bundle" not in st.session_state:
    loaded_model = load_pickle_from_gcs(bucket_name, MODEL_PATH)

    if loaded_model is None:
        loaded_model = new_model_bundle()
        st.info("No se encontro modelo previo en GCS. Se creo un modelo nuevo.")
    else:
        st.success("Modelo cargado desde GCS.")

    st.session_state.model_bundle = loaded_model

if "history" not in st.session_state:
    st.session_state.history = load_history_from_gcs(bucket_name, HISTORY_PATH)

if "files" not in st.session_state:
    st.session_state.files = []

if "file_index" not in st.session_state:
    st.session_state.file_index = 0


model_bundle = st.session_state.model_bundle


# =========================================================
# TABS
# =========================================================

tab_predict, tab_train_manual, tab_train_files, tab_model = st.tabs(
    [
        "Prediccion online",
        "Entrenar con caso manual",
        "Entrenamiento por archivo",
        "Estado del modelo"
    ]
)


# =========================================================
# TAB 1 - PREDICCION ONLINE
# =========================================================

with tab_predict:
    st.subheader("Prediccion online de severidad")

    col1, col2, col3 = st.columns(3)

    with col1:
        distance = st.number_input("Distance(mi)", value=0.5)
        temperature = st.number_input("Temperature(F)", value=70.0)
        humidity = st.number_input("Humidity(%)", value=60.0)
        visibility = st.number_input("Visibility(mi)", value=10.0)
        wind_speed = st.number_input("Wind_Speed(mph)", value=5.0)

    with col2:
        state = st.text_input("State", value="CA")
        weather = st.text_input("Weather_Condition", value="Clear")
        year_month = st.text_input("year_month", value="2023-03")
        hour = st.number_input("hour", min_value=0, max_value=23, value=18)
        month = st.number_input("month", min_value=1, max_value=12, value=3)

    with col3:
        dayofweek = st.number_input("dayofweek", min_value=0, max_value=6, value=2)
        is_weekend = st.selectbox("is_weekend", [0, 1], index=0)
        amenity = st.selectbox("Amenity", [0, 1], index=0)
        crossing = st.selectbox("Crossing", [0, 1], index=0)
        junction = st.selectbox("Junction", [0, 1], index=1)
        traffic_signal = st.selectbox("Traffic_Signal", [0, 1], index=0)

    x_manual = {
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

    if st.button("Predecir severidad"):
        y_pred, proba = predict_one(model_bundle, x_manual)

        if y_pred == 1:
            st.error("Prediccion: accidente severo")
        else:
            st.success("Prediccion: accidente no severo")

        st.write("Clase predicha:", y_pred)
        st.write("Probabilidades:", proba)


# =========================================================
# TAB 2 - ENTRENAMIENTO MANUAL
# =========================================================

with tab_train_manual:
    st.subheader("Prediccion y aprendizaje con etiqueta real")

    st.markdown(
        """
        Usa los mismos datos capturados en la pestana de prediccion.  
        Primero se predice, despues se aprende con la etiqueta real.
        """
    )

    y_true_manual = st.selectbox(
        "Etiqueta real: is_severe",
        [0, 1],
        index=0
    )

    if st.button("Predecir y entrenar con este caso"):
        y_pred, proba = predict_one(model_bundle, x_manual)

        learn_one(model_bundle, x_manual, int(y_true_manual))
        save_pickle_to_gcs(model_bundle, bucket_name, MODEL_PATH)

        st.session_state.model_bundle = model_bundle

        st.write("Prediccion previa:", y_pred)
        st.write("Etiqueta real aprendida:", int(y_true_manual))
        st.write("Probabilidades previas:", proba)
        st.success("Modelo actualizado y guardado en GCS.")


# =========================================================
# TAB 3 - ENTRENAMIENTO POR ARCHIVO
# =========================================================

with tab_train_files:
    st.subheader("Entrenamiento incremental por archivo mensual")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Listar archivos CSV"):
            files = list_csv_files(bucket_name, prefix)
            st.session_state.files = files
            st.session_state.file_index = 0
            st.success(f"Archivos encontrados: {len(files)}")

    with col_b:
        if st.button("Procesar siguiente archivo"):
            if not st.session_state.files:
                st.session_state.files = list_csv_files(bucket_name, prefix)
                st.session_state.file_index = 0

            files = st.session_state.files
            idx = st.session_state.file_index

            if idx >= len(files):
                st.success("Todos los archivos ya fueron procesados.")
            else:
                blob_name = files[idx]

                with st.spinner(f"Procesando {blob_name}"):
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

                st.success("Archivo procesado y modelo guardado.")
                st.json(result)

    with col_c:
        if st.button("Reiniciar modelo e historial"):
            delete_blob_if_exists(bucket_name, MODEL_PATH)
            delete_blob_if_exists(bucket_name, HISTORY_PATH)

            st.session_state.model_bundle = new_model_bundle()
            st.session_state.history = pd.DataFrame()
            st.session_state.files = []
            st.session_state.file_index = 0

            st.success("Modelo e historial reiniciados.")

    st.markdown("---")

    st.write("Archivo actual:", st.session_state.file_index)

    if st.session_state.files:
        st.write("Total de archivos listados:", len(st.session_state.files))

    if not st.session_state.history.empty:
        st.subheader("Historial de entrenamiento")

        hist_show = st.session_state.history.copy()
        st.dataframe(hist_show)

        chart_cols = [
            col
            for col in ["accuracy", "precision", "recall", "f1"]
            if col in hist_show.columns
        ]

        if chart_cols:
            st.line_chart(hist_show[chart_cols])


# =========================================================
# TAB 4 - ESTADO DEL MODELO
# =========================================================

with tab_model:
    st.subheader("Estado actual")

    st.write("Modelo:", model_bundle.get("model_name"))
    st.write("Creado en:", model_bundle.get("created_at"))
    st.write("Registros aprendidos:", model_bundle.get("rows_trained", 0))
    st.write("Ruta modelo GCS:", f"gs://{bucket_name}/{MODEL_PATH}")
    st.write("Ruta historial GCS:", f"gs://{bucket_name}/{HISTORY_PATH}")

    st.markdown("### Features usadas")
    st.write(SELECTED_FEATURES)

    st.markdown("### Nota")
    st.info(
        "Cloud Run no conserva archivos locales entre instancias. "
        "Por eso el modelo se guarda en Google Cloud Storage."
    )


st.caption("Cloud Run + Streamlit + River + Google Cloud Storage")