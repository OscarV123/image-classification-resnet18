from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from predict import load_model_and_preprocess, predict_image, build_resnet18
from utils_model import ensure_file
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
import torch, io
import os
import asyncio

MODEL_PATH = "artifacts/model.pt"
STATE_PATH = "artifacts/model_state.pt"
MODEL_URL = os.getenv("MODEL_URL")
CLASSES_PATH = "artifacts/class_names.json"
PREPROCESS_PATH = "artifacts/preprocess.json"

SEM = asyncio.Semaphore(4)
MAX_BYTES = 2 * 1024 * 1024
MAX_MB = MAX_BYTES // (1024 * 1024)
Image.MAX_IMAGE_PIXELS = 16_000_000

app = FastAPI(title="Resnet18 CIFAR-10 Image Classifier API", version="1.0.0")
app.state.ready_evt = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://oscarv123.github.io",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Accept"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = None
transform = None

@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    # exc.detail puede ser str o dict
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code") or f"HTTP_{exc.status_code}"
        message = exc.detail.get("message") or "Error"
    else:
        code = f"HTTP_{exc.status_code}"
        message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": code, "message": message},
    )

@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "code": "VALIDATION", "message": "Datos inválidos"},
    )

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    # No exponemos traceback al usuario
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": "INTERNAL", "message": "Error del servidor"},
    )

# carga del modelo
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
    ensure_loaded()
    # warmup
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(dummy)
    evt = getattr(app.state, "ready_evt", None)
    if evt is not None:
        evt.set()

@app.post("/predict")
async def predict_image_endpoint(request: Request, file: UploadFile = File(...)):

    try:
        cl = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        cl = 0
    if cl > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "FILE_TOO_LARGE", "message": f"La imagen supera el límite ({MAX_MB} MB)"},
        )

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "FILE_TOO_LARGE", "message": f"La imagen supera el límite ({MAX_MB} MB)"},
        )

    async with SEM:
        try:
            bio = io.BytesIO(data)
            img = Image.open(bio)
            img.verify()
            img = Image.open(io.BytesIO(data)).convert("RGB") 
        except (UnidentifiedImageError, DecompressionBombError):
            raise HTTPException(
                status_code=400,
                detail={"code": "BAD_IMAGE", "message": "El archivo no es una imagen válida"},
            )

        with torch.no_grad():
            label, probabilities = predict_image(img, transform, model, device, class_names)

        return {
            "status": "ok",
            "data": {
                "filename": file.filename,
                "prediction": label,
                "probabilities": probabilities
            },
        }
