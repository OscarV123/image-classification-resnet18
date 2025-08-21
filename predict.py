import torch
import torchvision.transforms as T
from torchvision import models
import json

def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

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

    state = torch.load(model_path, map_location=device)
    model = build_resnet18(len(class_names))
    model.load_state_dict(state)
    model.eval()

    return model, class_names, transform

def predict_image(img, transform, model, device, class_names):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
        index = int(torch.argmax(logits, dim=1))

    return class_names[index], dict(zip(class_names, probabilities))