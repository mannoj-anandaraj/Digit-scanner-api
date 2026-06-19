import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "mnist_cnn.h5")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def center_by_mass(arr):
    """
    Shifts the digit so its centre of mass sits at pixel (14,14) —
    the centre of a 28x28 image. This matches exactly how MNIST images
    were prepared during training, which is why it improves accuracy.
    """
    total = arr.sum()
    if total == 0:
        return arr
    rows = np.arange(arr.shape[0])
    cols = np.arange(arr.shape[1])
    cy = int((arr.sum(axis=1) * rows).sum() / total)
    cx = int((arr.sum(axis=0) * cols).sum() / total)
    shifted = np.roll(arr, 14 - cy, axis=0)
    shifted = np.roll(shifted, 14 - cx, axis=1)
    return shifted


def crop_and_prepare(arr):
    """
    Takes a (H, W) float array with white digit on black background.
    1. Crops tightly to the digit bounding box
    2. Resizes to 20x20 (MNIST digit size within the 28x28 frame)
    3. Places it centred on a 28x28 black canvas
    4. Centres by mass
    Returns (1, 28, 28, 1) float32 ready for the model.
    """
    mask = arr > 10
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)

    if not rows_any.any():
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]

    # Padding around bounding box
    pad = max(2, int(arr.shape[0] * 0.05))
    r0 = max(0, r0 - pad)
    r1 = min(arr.shape[0] - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(arr.shape[1] - 1, c1 + pad)

    cropped = arr[r0:r1 + 1, c0:c1 + 1]

    # Resize to 20x20 then place on 28x28 (matches MNIST framing)
    img = Image.fromarray(cropped.astype(np.uint8))
    img = img.resize((20, 20), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = np.array(img, dtype=np.float32)

    canvas = center_by_mass(canvas)
    return (canvas / 255.0).reshape(1, 28, 28, 1)


def segment_digits(arr):
    """
    Splits a (H, W) array into individual digit column regions using
    vertical projection — counts non-zero pixels per column, finds
    gaps between digit groups, and returns (col_start, col_end) pairs.
    """
    col_proj = (arr > 10).sum(axis=0).astype(float)

    # Smooth to reduce noise from thin strokes
    kernel = np.ones(5) / 5
    col_proj = np.convolve(col_proj, kernel, mode='same')

    in_digit = col_proj > 0.5

    segments = []
    start = None
    for i, v in enumerate(in_digit):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append([start, i])
            start = None
    if start is not None:
        segments.append([start, arr.shape[1]])

    if not segments:
        return segments

    # Merge segments that are too close (connected strokes, noise)
    min_gap = max(4, arr.shape[1] // 40)
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - merged[-1][1] < min_gap:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    return merged


def preprocess_image(image_bytes):
    """
    Opens image, auto-inverts if white background, segments digits,
    and returns a list of prepared (1, 28, 28, 1) arrays — one per digit.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Auto-invert: MNIST expects white digit on black background
    if arr.mean() > 127:
        arr = 255.0 - arr

    segments = segment_digits(arr)

    if len(segments) <= 1:
        # Single digit or unsegmentable — treat entire image as one digit
        return [crop_and_prepare(arr)]

    # Multiple digits — crop each column segment and prepare individually
    prepared = []
    for c0, c1 in segments:
        pad = max(2, (c1 - c0) // 10)
        c0p = max(0, c0 - pad)
        c1p = min(arr.shape[1], c1 + pad)
        prepared.append(crop_and_prepare(arr[:, c0p:c1p]))

    return prepared


def predict(image_bytes):
    """
    Runs inference on all detected digits.
    Returns number (full string), count, and per-digit predictions.
    """
    model = get_model()
    prepared = preprocess_image(image_bytes)

    predictions = []
    digits = []

    for arr in prepared:
        probs = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        scores = [round(float(p), 6) for p in probs]
        predictions.append({
            "digit": digit,
            "confidence": round(confidence, 6),
            "scores": scores
        })
        digits.append(str(digit))

    return {
        "number": "".join(digits),
        "count": len(predictions),
        "predictions": predictions
    }
