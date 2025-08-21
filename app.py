from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from predict import load_model_and_preprocess, predict_image, build_resnet18
from utils_model import ensure_file
from PIL import Image
import torch, io
import os

# Rutas necesarias antes de cargar el modelo
MODEL_PATH = "artifacts/model.pt"
STATE_PATH = "artifacts/model_state.pt"
MODEL_URL = os.getenv("MODEL_URL")
CLASSES_PATH = "artifacts/class_names.json"
PREPROCESS_PATH = "artifacts/preprocess.json"

app = FastAPI(title="Resnet18 CIFAR-10 Image Classifier API", version="1.0.0")

# CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = None
transform = None

def ensure_loaded():
    global model, class_names, transform
    if model is None:
        if MODEL_URL:
            ensure_file(MODEL_PATH, MODEL_URL)
        model, class_names, transform = load_model_and_preprocess(
            MODEL_PATH, CLASSES_PATH, PREPROCESS_PATH, device, STATE_PATH, build_resnet18
        )

@app.on_event("startup")
def on_startup():
    print("App started. Modelo aún no cargado (lazy load).")

@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "starting",
        "device": str(device),
        "num_classes": len(class_names or [])
    }

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    ensure_loaded()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        label, probabilities = predict_image(img, transform, model, device, class_names)
    return {"filename": file.filename, "prediction": label, "probabilities": probabilities}
