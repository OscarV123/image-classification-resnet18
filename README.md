# ResNet18 CIFAR-10 Image Classifier | Full-Stack ML Web App
A complete full-stack image classification web application that uses transfer learning and fine-tuning on a pre-trained ResNet18 model to classify the 10 classes of the CIFAR-10 dataset

**Live Demo -> [https://oscarv123.github.io/image-classification-resnet18/]**

## Project Structure
- `artifacts/`: Saved model files and metadata (`model.pt`, `model_state.pt`, `class_names.json`, `preprocess.json`).
- `docs/`: Frontend files (web page: HTML, CSS, and JavaScript)
- `images_input/`: Sample test images (feel free to replace them with your own)
- `notebook/`: Training notebook (`Ejercicio de transfer learning y finetuning.ipynb`).
- `.gitignore`: Files and folders ignored by Git.
- `requirements.txt`: Python dependencies for the project. Install with: *pip install -r requirements.txt*
- `app.py`: Backend API endpoints and HTTP error handling.
- `LICENSE`: Project license.
- `main.py`: Main script to run the application (start here after choosing images for `images_input/`).
- `predict.py`: Prediction functions and model loading logic.
- `utils_model.py`: Helper functions for model loading.

## Notebook Details & Experiments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/OscarV123/image-classification-resnet18/blob/main/notebook/Ejercicio_de_transfer_learning_y_finetuning.ipynb)
Inside the file containing all the logic of the **AI** (`Ejercicio de transfer learning y finetuning.ipynb`) I've created three models:
1. **Multilayer Perceptron (MLP)** - Machine Learning
2. **ResNet18 -> Transfer Learning** - Deep Learning
3. **ResNet18 -> Transfer Learning + Fine-Tuning (4th layer)** - Deep Learning

All models were trained, validated and evaluated using CIFAR-10 dataset.

**¡FOR THE FUNCTIONS OF `predict.py`, I USED THE BEST-PERFORMING MODEL (*ResNet18 -> Transfer Learning + Fine-Tuning (4th layer)*)!**

The main objective of training three different architectures was to compare their performance on the same task. To do this, I analyzed:

*Graphics*
- **Training curves** (loss and accuracy) over epoch.
- **Confusion matrix** to analyze the model's ability to correctly classify each class.

*Metrics over epoch*
- **Training accuracy**
- **Validation accuracy**
- **The Best validation accuracy**
- **Test accuracy**

## Required Python Version
- Python >= 3.13.1

## Steps to run the project locally
### 1. Download the .zip file or clone the repository and open it with your favorite code editor
You can download it by click on de "<>code" button at the top of the repository.

### 2. Install the necessary libraries
On a terminal, type the following command: *pip install -r requirements.txt*

### 3. Run main.py
Do this from your code editor or by typing in the terminal: *python main.py*

# ¡DONE!
