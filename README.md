# Brain Tumor Segmentation with UNet

## Project Overview

This project implements a successful **UNet-based deep learning model** for **brain tumor segmentation** using MRI scans. Achieved a **test accuracy of 98.31%** after hyperparameter fine-tuning and architecture optimizations.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Project Structure](#project-structure)

---

## Installation

Clone the repo:

      git clone https://github.com/vnieder/UNet-BrainMRI-Segmentation.git

Install Dependencies:

```bash
cd UNet-BrainMRI-Segmentation
pip install -r requirements.txt
```

If running on Google Colab or Kaggle notebook, most core dependencies should already be installed.

---

## Dataset Preparation

Dataset:

      https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data

The dataset is split into **training data**, **validation data**, and **test data** sets.

- Training data is used to learn key features from MRI images.
- Validation data is used to monitor performance during testing.
- Test data is unforeseen data used to evaluate the final model after training.

Annotations are in **COCO format**. The preprocessing pipeline (see dataloader.py) converts the images and masks into TensorFlow tensors.

---

## Model Architecture

Original UNet architecture paper: https://arxiv.org/pdf/1505.04597

     ![UNet Architecture](images/UNet_Model.jpg)

In short, UNet consists of two main paths:

- Encoder: this path captures context through a series of downsampling and convolutional layers.

- Decoder: this path enables precise localization by upsampling and concatenating features from the Encoder through skip connections.

These two paths are followed by a final layer of 1x1 convolutions with a sigmoid activation function to make predictions on each pixel.

### Improvements to Base UNet

- Dropout layers to reduce overfitting
- Batch Normalization to stabilize and accelerate training

### Loss Function (weighted composite)

```python
Loss = 0.4 * Binary Cross Entropy + 0.4 * Dice Loss + 0.2 * IoU Loss
```

---

## Results

Final test accuracy: **98.37%**

[TODO: add pictures here]

---

## Project Structure

```
UNet-BrainMRI-Segmentation/
├── src/
│   ├── criterion/       # Loss functions
│   ├── dataloader/      # Dataset and preprocessing
│   ├── model/           # UNet architecture
│   ├── test/            # Evaluation scripts
│   ├── training/        # Training logic
│   └── viz/             # Visualization utilities
├── images/              # Output images and metrics
├── requirements.txt
├── .gitignore
└── README.md
```
