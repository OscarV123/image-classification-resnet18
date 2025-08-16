# Transfer Learning con ResNet18 en CIFAR-10
Proyecto de entrenamiento de un modelo ResNet18 usando Transfer Learning y Fine-Tuning para clasificar imágenes del dataset CIFAR-10.

## Estructura del proyecto
- `artifacts/`: modelos y metadatos exportados (`model.pt`, `model_state.pt`, `class_names.json`, `preprocess.json`).
- `images_input/`: algunas imagenes de prueba que deje, reemplázalas con las tuyas si deseas.
- `notebook/`: notebook de entrenamiento (`Ejercicio de transfer learning y finetuning.ipynb`).
- `requirements.txt`: dependencias necesarias para correr el proyecto. Descargalas con: *pip install -r requirements.txt*
- `.gitignore`: exclusiones de Git.
- `main.py`: único .py a ejecutar luego de revisar `images_input/`.
- `predict.py`: funciones de predicción y carga de modelo.


## Detalles sobre el notebook
Dentro del archivo que contiene toda la lógica de la **Inteligencia Artificial** (`Ejercicio de transfer learning y finetuning.ipynb`) he creado tres modelos:
1. **Perceptrón multicapa (MLP)** - Machine Learning
2. **Modelo con transfer** learning de ResNet18 - Deep Learning
3. **Modelo con transfer learning y fine-tuning** de la capa layer4 de ResNet18 - Deep Learning

Todos los modelos fueron entrenados, validados y evaluados utilizando el dataset CIFAR-10.

**¡PARA LAS FUNCIONES DE `predict.py` USE EL MODELO CON MEJOR RENDIMIENTO (*Modelo con transfer learning y fine-tuning*)!**

Mi objetivo al crear tres arquitecturas distintas es comparar su rendimiento individual. Para ello, utilice tres tipos de gráficos y otras metricas: 
*Gráficos*
- **Curvas de entrenamiento** (pérdida y precisión) en función de las épocas.
- **Matriz de confusión** para analizar la capacidad del modelo de clasificar correctamente cada clase.
*Metricas durante las fases principales*
- **Precisión en entrenamiento**
- **Precisión en validación**
- **Mejor precisión en validación**
- **Precisión en prueba**

## Requisitos
- Python 3.13.1
- torch==2.8.0
- torchvision==0.23.0
- matplotlib==3.10.5
- scikit-learn==1.7.1
- numpy==2.2.1