from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.schemas import PredictionResponse, HealthResponse
from app.model import get_model, predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts — not on each request
    logger.info("Loading MNIST CNN model...")
    get_model()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="MNIST Digit Classifier API",
    description=(
        "A REST API that classifies handwritten digits (0–9) using a "
        "Convolutional Neural Network trained on the MNIST dataset.\n\n"
        "**Model:** 2-block CNN (Conv→MaxPool×2 → Dense → Dropout → Softmax)\n"
        "**Accuracy:** ~99.2% on MNIST test set\n"
        "**Author:** Mannoj Anandaraj | MSc Advanced Computing, King's College London"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint. Returns API status and model info.
    Use this to confirm the service is running before sending predictions.
    """
    return {
        "status": "ok",
        "model": "MNIST CNN (TensorFlow/Keras)",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_digit(file: UploadFile = File(...)):
    """
    Upload a handwritten digit image and get the predicted digit.

    - **file**: PNG, JPEG, or BMP image of a handwritten digit (ideally white digit on black background, like MNIST).
    - **Returns**: predicted digit (0–9), confidence score, and softmax scores for all 10 classes.
    """
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use PNG, JPEG, or BMP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check that the image is valid.")

    return result
