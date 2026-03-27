# 🧠 Medbrain-net: a dual-pathway framework for brain tumor segmentation and classification from multi-modal MR images

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

## 📌 Project Overview
**MedBrain-Net** is an advanced, automated medical imaging framework designed to assist radiologists in the initial diagnosis and treatment planning of brain tumors. By merging two specialized deep learning pathways, the system provides a comprehensive clinical profile of the tumor:

* **Precise Segmentation:** High-fidelity boundary delineation using attention-gated architectures.
* **Advanced Classification:** Intelligent subtype identification (Glioma, Meningioma, Pituitary).

> **Clinical Impact:** This framework effectively minimizes inter-observer variation and accelerates diagnostic workflows, delivering stable and interpretable results in high-pressure clinical settings.

---

## 🏗️ System Architecture

MedBrain-Net operates via a dual-branch logic to ensure both spatial accuracy and categorical precision:

<img width="1126" height="783" alt="image" src="https://github.com/user-attachments/assets/efecf523-9e38-455f-9239-c5f07ab4d062" />

### 1. 🟥 Segmentation Branch (PyTorch)
An ensemble of **U-Net++** models optimized for the BraTS 2020 dataset.
* **Nested Skip Connections:** Captures multi-scale features for complex tumor geometries.
* **CBAM (Convolutional Block Attention Module):** Dynamically suppresses noise and enhances tumor-relevant feature maps.
* **Deep Supervision:** Ensures gradient flow and feature consistency across all decoding levels.
* **Hybrid Loss:** Optimized via $Loss = \text{BCE} + \text{Dice} + \text{Focal Loss}$ to handle severe class imbalance.

### 2. 🟦 Classification Branch (TensorFlow/Keras)
A **DenseNet201** backbone utilizing transfer learning on the SARTAJ dataset.
* **Feature Re-use:** Dense blocks ensure maximum information flow for subtype differentiation.
* **Robust Regularization:** Implements a 0.5 Dropout rate to prevent overfitting on specific scan orientations.
* **On-the-fly Augmentation:** Integrated Keras layers for rotation, zoom, and horizontal flipping.

---

## 📊 Performance Benchmarks

| Metric | Segmentation (BraTS 2020) | Classification (SARTAJ) |
| :--- | :---: | :---: |
| **Accuracy** | — | **97.25%** |
| **F1-Score** | - | **95%** |
| **Dice Coefficient** | **0.8237** | — |
| **Mean IoU** | **0.7051** | — |

