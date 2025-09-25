# 🐾 Animals10 Image Classification

This project trains a **Convolutional Neural Network (CNN)** to classify images of animals into **10 categories** using the [Animals10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

---

## 📂 Dataset

- Source: Kaggle Animals10  
- Classes included:  
  ```
  dog, cat, horse, spider, butterfly,
  chicken, cow, sheep, squirrel, elephant
  ```
- Images are resized and augmented before training.

---

## ⚙️ Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/animals10-classification.git
cd animals10-classification
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` should include:
```
torch
torchvision
matplotlib
seaborn
scikit-learn
numpy
```

---


- **Model**: CNN (PyTorch)  
- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam  
- **Data Augmentation**: random flips, rotations, normalization  

---

## 📊 Evaluation

The model is evaluated using:

- ✅ **Accuracy**
- ✅ **Confusion Matrix**
- ✅ **Classification Report** (Precision, Recall, F1-score)
- ✅ **Loss & Accuracy Curves**
- ✅ **Misclassified Images Analysis**

### Training Curves
![Training Curves](results/training_curves.png)

### Example Misclassifications
Images are shown with **True vs Predicted labels** to analyze where the model fails.

---

## 🔮 Future Work

- Fine-tune with **ResNet/EfficientNet**
- Handle class imbalance (oversampling or class weights)
- Perform hyperparameter tuning

---

## 🙌 Acknowledgments

- Dataset: [Kaggle Animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)  
- Framework: [PyTorch](https://pytorch.org/)

---
