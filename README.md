# MNIST Digit Classifier API

A production-ready REST API that classifies handwritten digits (0–9) using a Convolutional Neural Network trained on the MNIST dataset. Built with FastAPI and TensorFlow/Keras, containerised with Docker, and deployed on Render.

**Live API →** `https://digit-scanner-api.onrender.com` *(replace with your Render URL after deployment)*  
**Interactive Docs →** `https://digit-scanner-api.onrender.com/docs`

---

## Model

| Property | Detail |
|---|---|
| Architecture | 2-block CNN: Conv(32)→MaxPool → Conv(64)→MaxPool → Dense(128) → Dropout → Softmax(10) |
| Dataset | MNIST (60,000 train / 10,000 test) |
| Test Accuracy | ~99.2% |
| Framework | TensorFlow / Keras |
| Input | 28×28 grayscale image |
| Output | Predicted digit (0–9), confidence score, softmax scores for all 10 classes |

Trained as part of MSc Advanced Computing coursework at King's College London.

---

## API Endpoints

### `GET /`
Health check — confirms the service is running and the model is loaded.

```json
{
  "status": "ok",
  "model": "MNIST CNN (TensorFlow/Keras)",
  "version": "1.0.0"
}
```

### `POST /predict`
Upload a handwritten digit image and get the predicted digit.

**Request:** `multipart/form-data` with a `file` field (PNG, JPEG, or BMP).

**Response:**
```json
{
  "digit": 7,
  "confidence": 0.9966,
  "scores": [0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.0, 0.997, 0.0, 0.0]
}
```

---

## Run Locally

**Prerequisites:** Python 3.11+

```bash
# Clone the repo
git clone https://github.com/mannoj-anandaraj/mnist-api.git
cd mnist-api

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Run with Docker

```bash
# Build the image
docker build -t mnist-api .

# Run the container
docker run -p 8000:8000 mnist-api
```

---

## Project Structure

```
mnist-api/
├── app/
│   ├── main.py          # FastAPI routes and lifespan
│   ├── model.py         # Model loading and inference pipeline
│   └── schemas.py       # Pydantic request/response schemas
├── model/
│   └── mnist_cnn.h5     # Trained Keras model weights
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — async REST framework with auto-generated Swagger docs
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — model training and inference
- **[Pillow](https://pillow.readthedocs.io/)** — image preprocessing (resize, grayscale, normalise)
- **[Docker](https://www.docker.com/)** — containerisation
- **[Render](https://render.com/)** — cloud deployment (free tier)

---

## Author

**Mannoj Anandaraj**  
MSc Advanced Computing, King's College London (2025–2026)  
[GitHub](https://github.com/mannoj-anandaraj) · [LinkedIn](https://linkedin.com/in/mannoj-anandaraj)
