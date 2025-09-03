# Transfer Learning con ResNet18 en CIFAR-10
Proyecto de entrenamiento de un modelo ResNet18 usando Transfer Learning y Fine-Tuning para clasificar imágenes relacionadas al dataset CIFAR-10.

## Estructura del proyecto
- `artifacts/`: modelos y metadatos exportados (`model.pt`, `model_state.pt`, `class_names.json`, `preprocess.json`).
- `docs/`: archivos relacionados al frontend (página web).
- `images_input/`: algunas imágenes de prueba que dejé, reemplázalas con las tuyas si deseas.a 
- `notebook/`: notebook de entrenamiento (`Ejercicio de transfer learning y finetuning.ipynb`).
- `.gitignore`: exclusiones de Git.
- `requirements.txt`: dependencias necesarias para correr el proyecto. Descárgalas con: *pip install -r requirements.txt*
- `app.py`: funcionalidades del backend endpoint y manejo de errores HTTP.
- `LICENSE`: derechos del proyecto.
- `main.py`: único .py a ejecutar luego de revisar `images_input/`.
- `predict.py`: funciones de predicción y carga de modelo.
- `utils_model.py`: carga auxiliar del modelo.

## Detalles sobre el notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/OscarV123/image-classification-resnet18/blob/main/notebook/Ejercicio_de_transfer_learning_y_finetuning.ipynb)
Dentro del archivo que contiene toda la lógica de la **Inteligencia Artificial** (`Ejercicio de transfer learning y finetuning.ipynb`) he creado tres modelos:
1. **Perceptrón multicapa (MLP)** - Machine Learning
2. **Modelo con transfer learning de ResNet18** - Deep Learning
3. **Modelo con transfer learning y fine-tuning** de la capa layer4 de ResNet18 - Deep Learning

Todos los modelos fueron entrenados, validados y evaluados utilizando el dataset CIFAR-10.

**¡PARA LAS FUNCIONES DE `predict.py` USÉ EL MODELO CON MEJOR RENDIMIENTO (*Modelo con transfer learning y fine-tuning*)!**

Mi objetivo al crear tres arquitecturas distintas es comparar su rendimiento individual. Para ello, utilicé tres tipos de gráficos y otras métricas: 

*Gráficos*
- **Curvas de entrenamiento** (pérdida y precisión) en función de las épocas.
- **Matriz de confusión** para analizar la capacidad del modelo de clasificar correctamente cada clase.

*Métricas durante las fases principales*
- **Precisión en entrenamiento**
- **Precisión en validación**
- **Mejor precisión en validación**
- **Precisión en prueba**

## Versión de Python necesaria
- Python >= 3.13.1

## Pasos para ejecutar el proyecto en local
### 1. Descargar el .zip o clonar el repositorio y abrir en tu editor de código
Puedes descargarlo en el botón de "<>code" al inicio del repositorio.

### 2. Instalar las librerías necesarias
En un cmd escribe lo siguiente: *pip install -r requirements.txt*

### 3. Ejecuta main.py
Hazlo desde tu editor de código o escribiendo en el cmd: *python main.py*

# ¡LISTO!
