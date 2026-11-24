import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Reciclaje Computacion en la nube",
    layout="centered"
)

st.title(
    "Modelo predictivo Reciclaje clase de IA-UTH - 2026 - Josue Ventura"
)

st.write(
    "Suba una imagen para clasificar con el modelo MobileNetV2 pre entrenado"
)


IMG_SIZE = (224, 224)

MODEL_DIR = Path(__file__).resolve().parent / "modelo_reciclaje_mobilenet"

CLASSES_PATH = MODEL_DIR / "class_names.json"
MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"


LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


def load_class_names() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error(f"No se encontró el archivo de clases: {CLASSES_PATH}")
        st.stop()

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    model_path = None

    if MODEL_PATH_H5.exists():
        model_path = MODEL_PATH_H5
    elif MODEL_PATH_KERAS.exists():
        model_path = MODEL_PATH_KERAS

    if model_path is None:
        st.error(
            "No se encontró el modelo. Sube al repositorio la carpeta "
            "'modelo_reciclaje_mobilenet' con 'waste_mobilenet.h5' "
            "o 'waste_mobilenet.keras' y 'class_names.json'."
        )
        st.stop()

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()
