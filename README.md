# UNet Segmentation Project

## Project Overview
UNet is a convolutional neural network specifically developed for image segmentation tasks. 



---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architecture](#model-architecture)
5. [Project Structure](#project-structure)

---

## Installation

To get started, fork the repository and install the required dependencies.
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo

```bash
cd UNet_Project
pip install -r requirements.txt
```
---

## Dataset Preparation

The dataset is split into training data, validation data, and test data. The segmentation model is supposed to learn the features in the images from the training data, and the model performance is monitored on the validation data after each epoch of learning. Once the model weights are finalized, the model performance is quantified using the test data. 

Prepare the segmentation masks using the COCO format annotation JSON file as outlined in "UNet_Project_Description.ipynb". Develop a pipeline to convert the jpg images to TensorFlow tensors (along with appropriate masks) as inputs to the UNet model.

## Model Architecture
UNet consists of two main paths:

Contracting Path: Also known as the encoder, this path captures context through a series of downsampling and convolutional layers.

Expansive Path: Also known as the decoder, this path enables precise localization by upsampling and concatenating features from the encoder.

These two paths are followed by a final layer of 1x1 (or depthwise) convolutions to make predictions on each pixel. Learn more about the UNet architecture here,
https://arxiv.org/abs/1505.04597

---

## Project Structure
