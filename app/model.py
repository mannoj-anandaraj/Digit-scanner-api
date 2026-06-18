import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "mnist_cnn.h5")

# Load once at startup — not on every request
_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Accepts any common image format (PNG, JPEG, BMP).
    Converts to 28x28 grayscale, normalises to [0, 1],
    and reshapes to (1, 28, 28, 1) for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict(image_bytes: bytes) -> dict:
    """
    Run inference and return digit, confidence, and all class scores.
    """
    model = get_model()
    arr = preprocess_image(image_bytes)
    probs = model.predict(arr, verbose=0)[0]          # shape: (10,)
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    scores = [round(float(p), 6) for p in probs]
    return {"digit": digit, "confidence": round(confidence, 6), "scores": scores}
