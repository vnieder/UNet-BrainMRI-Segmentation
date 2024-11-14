# UNet Segmentation Project

## Project Overview
UNet is a convolutional neural network specifically developed for image segmentation tasks. Develop a UNet-based model to detect tumor regions in brain MRI images.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)

---

## Installation

If developing on a local system, fork the repository and install the required dependencies.
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo

```bash
cd UNet_Project
pip install -r requirements.txt
```

If developing on Google Colab or Kaggle notebook, most core dependencies should already be installed. Install additional dependencies via pip.
```python
!pip install tqdm
```

---

## Dataset Preparation

The dataset is split into training data, validation data, and test data. The segmentation model is supposed to learn the features in the images from the training data, and the model performance is monitored on the validation data after each epoch of learning. Once the model weights are finalized, the model performance is quantified using the test data. 

Prepare the segmentation masks using the COCO format annotation JSON file as outlined in "UNet_Project_Description.ipynb". Develop a pipeline to convert the jpg images to TensorFlow tensors (along with appropriate masks) as inputs to the UNet model.

---

## Model Architecture
UNet consists of two main paths:

Contracting Path: Also known as the encoder, this path captures context through a series of downsampling and convolutional layers.

Expansive Path: Also known as the decoder, this path enables precise localization by upsampling and concatenating features from the encoder.

These two paths are followed by a final layer of 1x1 (or depthwise) convolutions to make predictions on each pixel. Learn more about the UNet architecture here,
https://arxiv.org/abs/1505.04597

---

## Project Directory Structure
```
UNet_Project/  
 
├── src/ # Source code 
│ ├── criterion/ # Loss function 
│ ├── dataloader/ # Data preparation 
│ ├── model/ # Model architecture 
│ ├── test/ # Testing tools 
│ ├── training/ # Training procedure (training loop, optimizer, scheduler) 
│ └── viz/ # Visualization tools 
├── results/ 
├── UNet_Project_Description.ipynb # Project description notebook 
├── requirements.txt # Python dependencies 
└── README.md # Project README 
```