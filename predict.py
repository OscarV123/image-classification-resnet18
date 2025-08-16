import torch
import torchvision.transforms as T
from PIL import Image
import json

def load_model_and_preprocess(model_path, classes_path, preprocess_path, device):
    # Cargar clases
    with open(classes_path) as f:
        class_names = json.load(f)

    # Cargar parámetros de preprocesamiento
    with open(preprocess_path) as f:
        preprocess_params = json.load(f)

    transform = T.Compose([
        T.Resize(preprocess_params['resize']),
        T.ToTensor(),
        T.Normalize(preprocess_params['normalize_mean'], preprocess_params['normalize_std'])
    ])

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    return model, class_names, transform

def predict_image(path, transform, model, device, class_names):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
        index = int(torch.argmax(logits, dim=1))

    return class_names[index], dict(zip(class_names, probabilities))