from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from predict import load_model_and_preprocess, predict_image
from PIL import Image
import torch, io
import os

# Rutas necesarias antes de cargar el modelo
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
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
        model, class_names, transform = load_model_and_preprocess(
            MODEL_PATH, CLASSES_PATH, PREPROCESS_PATH, device
        )

@app.on_event("startup")
def load_assets():
    global model, class_names, transform
    
    model, class_names, transform = load_model_and_preprocess(MODEL_PATH, 
                                                              CLASSES_PATH,
                                                              PREPROCESS_PATH,
                                                              device)

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "device": str(device),
        "num_classes": len(class_names or [])
    }

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    ensure_loaded()
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    label, probabilities = predict_image(img, transform, model, device, class_names)

    return {
        "filename": file.filename,
        "prediction": label,
        "probabilities": probabilities    
    }