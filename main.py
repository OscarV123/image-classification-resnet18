import torch
import json
from PIL import Image
import os
from predict import load_model_and_preprocess, predict_image

def print_predictions(input_dir="images_input", transform=None, model=None, device=None, class_names=None):
    
    def _print_one(img_path, transform, model, device):
        
        file = os.path.basename(img_path)
        label, probabilities = predict_image(img_path, transform, model, device, class_names)

        items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True) # Ordenar las probabilidades de manera descendente

        max_class = items[0][0] # clase con mayor probabilidad

        print(f"Archivo: {file}")
        print(f"Predicción: {label}")
        print("Probabilidades:")
        for clase, prob in items:
            print(f" {clase:>10}: {prob:.4f}")
        print("-" * 40)

    extensiones = (".png", ".jpg", ".jpeg")
    
    if os.path.isdir(input_dir):
        for file in sorted(os.listdir(input_dir)):
            if file.lower().endswith(extensiones):
                _print_one(os.path.join(input_dir, file), transform, model, device)
                
    elif os.path.isfile(input_dir) and input_dir.lower().endswith(extensiones):
        _print_one(input_dir, transform, model, device)
        
    else:
        print(f"Ruta no válida: {input_dir}")
    
    
if __name__ == "__main__":
    # Rutas necesarias antes de cargar el modelo
    MODEL_PATH = "artifacts/model.pt"
    CLASSES_PATH = "artifacts/class_names.json"
    PREPROCESS_PATH = "artifacts/preprocess.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, transform = load_model_and_preprocess(MODEL_PATH, CLASSES_PATH, PREPROCESS_PATH, device)

    print_predictions(transform=transform, model=model, device=device, class_names=class_names)  # Llamar a la función para imprimir las predicciones
