import torch
import torchvision.transforms as T
from torchvision import models
import json

def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model_and_preprocess(
    model_path: str,            # TorchScript: artifacts/model.pt
    classes_path: str,          # artifacts/class_names.json
    preprocess_path: str,       # artifacts/preprocess.json
    device: torch.device,
    state_path: str | None = None,  # artifacts/model_state.pt (opcional)
    build_model_fn=None,            # función que construya la arquitectura para state_dict
):
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(preprocess_path, "r", encoding="utf-8") as f:
        preprocess_params = json.load(f)

    transform = T.Compose([
        T.Resize(preprocess_params["resize"]),
        T.ToTensor(),
        T.Normalize(preprocess_params["normalize_mean"], preprocess_params["normalize_std"]),
    ])

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model, class_names, transform
    except Exception as e:
        print(f"Error al cargar TorchScript: {e}")
        print("Intentando cargar como state_dict...")

    if not state_path:
        raise RuntimeError("No se proporcionó 'state_path' para cargar el state_dict.")

    if build_model_fn is None:
        raise RuntimeError("Se requiere 'build_model_fn(num_classes)' para reconstruir la arquitectura.")

    state = torch.load(state_path, map_location=device)

    if isinstance(state, dict) and any(k in state for k in ("state_dict", "model_state_dict")):
        state = state.get("state_dict", state.get("model_state_dict"))

    if isinstance(state, dict) and len(state) and next(iter(state)).startswith("model."):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    model = build_model_fn(len(class_names))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[state_dict] Claves faltantes (primeras): {missing[:5]}")
    if unexpected:
        print(f"[state_dict] Claves inesperadas (primeras): {unexpected[:5]}")

    model.to(device).eval()
    return model, class_names, transform


def predict_image(img, transform, model, device, class_names):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
        index = int(torch.argmax(logits, dim=1))

    return class_names[index], dict(zip(class_names, probabilities))