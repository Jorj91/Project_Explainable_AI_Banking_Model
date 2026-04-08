# 🏦 Explainable AI for Signature Verification in Banking

## 📌 Project Overview

This project implements an **Explainable AI (XAI)** system for handwritten signature verification in a banking context. The goal is to classify signatures as **genuine or forged** while providing transparent explanations to support fraud detection, auditing, and regulatory compliance.

A **DenseNet deep learning model** is trained for high-accuracy classification, and multiple **XAI techniques** are applied to explain predictions. Additionally, a fully interpretable **Decision Tree** is built using handcrafted features to compare transparency vs performance.

## 📊 Dataset

The project uses the **CEDAR handwritten signature dataset** from Kaggle.

Dataset characteristics:

- 55 writers
- 24 genuine signatures per writer
- 24 forged signatures per writer
- Total images: 2640
- Balanced dataset (50% genuine / 50% forged)

This dataset reflects real banking scenarios such as:

- ✍️ contract signature verification
- 💳 payment authorization
- 🪪 identity validation
- 🚨 fraud detection

A **subject-level split** is used (train/val/test) to avoid writer leakage and ensure proper generalization.


## 🧠 Methodology

**Deep Learning Model - DenseNet**
- DenseNet-121 (transfer learning)
- Binary classifier (genuine vs forged)
- Data augmentation on training set
- Early stopping + best model checkpoint
**Test Accuracy: 94.9%**


**Explainable System - Decision Tree**

A transparent Decision Tree is trained using interpretable features:

- stroke thickness
- curvature
- pixel density
- aspect ratio
**Test accuracy: 59%**
  
This highlights the trade-off between interpretability and performance.

## 🔍 Explainable AI (XAI)
**Local explanations**
- LIME (model-agnostic)
- Grad-CAM
- Integrated Gradients
- Occlusion
- SHAP

These methods highlight **signature regions influencing predictions**, showing that the model relies mainly on **stroke structure** and **handwriting patterns**.

**Global explanations**
- Global SHAP
- Global Grad-CAM

Global attribution maps identify dataset-level importance regions and confirm that predictions are based on **signature strokes rather than background**.

## ⚙️ Pipeline

The project follows a **modular** pipeline where most logic is implemented in the src/ folder:

```text
src/
├── data.py            # dataset download, split, dataloaders
├── model.py           # DenseNet architecture
├── train.py           # training loop + early stopping
├── evaluation.py      # confusion matrix + metrics
├── explainability.py  # XAI utilities and visualization
```


**Pipeline steps:**

1. Dataset download (Kaggle)
2. Subject-level data split
3. Data preprocessing & dataloaders
4. DenseNet training
5. Model evaluation
6. Local XAI explanations
7. Global XAI explanations
8. Interpretable Decision Tree model

This modular design allows **easy reuse, debugging, and experimentation.**

## 🚀 How to Run

Open the notebook in Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jorj91/Project_Explainable_AI_Banking_Model/blob/main/main.ipynb)

Or view the notebook on GitHub: [main.ipynb](https://github.com/Jorj91/Project_Explainable_AI_Banking_Model/blob/main/main.ipynb)


## 📈 Results

| Model         | Accuracy | Interpretability    |
| ------------- | -------- | ------------------- |
| DenseNet      | 94.9%    | XAI-based           |
| Decision Tree | 59%      | Fully interpretable |

The project demonstrates that deep learning provides strong performance, while XAI techniques ensure transparency, making the system suitable for banking fraud detection and regulatory compliance.
