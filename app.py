import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# Configuración de la página
st.set_page_config(
    page_title="Reciclaje Computación en la nube",
    layout="centered"
)

st.title(
    "Modelo predictivo de reciclaje "
    "IA-UTH - 2026 - Josue Ventura"
)

st.write(
    "Sube una imagen para clasificarla con tu modelo "
    "de reciclaje basado en MobileNetV2."
)


# Rutas y configuración
IMG_SIZE = (224, 224)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelo_reciclaje_mobilenet"

CLASSES_PATH = MODEL_DIR / "class_names.json"
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"
MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


def load_class_names() -> list[str]:
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de clases: {CLASSES_PATH}"
        )

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) for name in class_names)
    ):
        raise ValueError(
            "class_names.json debe contener una lista no vacía "
            "de nombres de clases."
        )

    return class_names


@st.cache_resource
def load_model():
    # Si existen ambos archivos, se utiliza primero el formato .keras.
    for model_path in (MODEL_PATH_KERAS, MODEL_PATH_H5):
        if model_path.is_file():
            return tf.keras.models.load_model(
                str(model_path),
                compile=False
            )

    raise FileNotFoundError(
        "No se encontró el modelo. La carpeta "
        "'modelo_reciclaje_mobilenet' debe contener "
        "'waste_mobilenet.keras' o 'waste_mobilenet.h5'."
    )


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model, class_names, image):
    prepared_image = prepare_image(image)
    output = model.predict(prepared_image, verbose=0)
    predictions = np.asarray(output)

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(
            "Se esperaba una salida del modelo con forma "
            "(1, número_de_clases), pero se obtuvo "
            f"{predictions.shape}."
        )

    predictions = predictions[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            f"El modelo devuelve {len(predictions)} valores, "
            f"pero class_names.json contiene {len(class_names)} clases."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "El modelo devolvió valores no válidos en la predicción."
        )

    # La interfaz de porcentajes requiere probabilidades.
    if (
        np.any(predictions < 0)
        or np.any(predictions > 1)
        or not np.isclose(predictions.sum(), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "La salida del modelo no parece contener probabilidades "
            "que sumen 1. Revisa la activación de salida del modelo "
            "antes de mostrar estos valores como porcentajes."
        )

    return predictions


# Cargar clases y modelo
st.caption(
    "Clases: cartón, vidrio, metal, papel, plástico y basura general."
)

try:
    class_names = load_class_names()

    with st.spinner("Cargando el modelo..."):
        model = load_model()

except Exception as error:
    st.error(f"No se pudo iniciar el clasificador: {error}")
    st.stop()


# Subir imagen
uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Carga una imagen para comenzar.")

else:
    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = uploaded_image.convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        st.error(f"No se pudo abrir la imagen: {error}")
        st.stop()

    st.image(
        image,
        caption="Imagen cargada",
        width=400
    )

    try:
        with st.spinner("Analizando la imagen..."):
            predictions = predict_image(
                model,
                class_names,
                image
            )

    except Exception as error:
        st.error(f"No se pudo analizar la imagen: {error}")
        st.stop()

    # Ordenar de mayor a menor puntuación
    top_indices = np.argsort(predictions)[::-1][:3]

    main_index = int(top_indices[0])
    main_class = class_names[main_index]
    main_label = LABELS_ES.get(main_class, main_class)
    main_score = float(predictions[main_index]) * 100

    st.subheader("Predicción principal")
    st.success(f"{main_label}: {main_score:.2f}%")

    st.subheader("Top 3")

    for index in top_indices:
        raw_label = class_names[int(index)]
        label = LABELS_ES.get(raw_label, raw_label)
        score = float(predictions[index]) * 100

        st.write(f"- {label}: {score:.2f}%")

    st.subheader("Detalle por clase")

    for index, raw_label in enumerate(class_names):
        label = LABELS_ES.get(raw_label, raw_label)
        score = float(predictions[index])

        st.progress(
            min(1.0, max(0.0, score)),
            text=f"{label}: {score * 100:.2f}%"
        )

    st.caption(
        "Los porcentajes corresponden a la salida del modelo; "
        "no garantizan que la clasificación sea correcta."
    )
