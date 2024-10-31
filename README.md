# Diffusion-Based Object Detection Pipeline


<img width="1417" alt="Screenshot 2024-10-30 at 4 54 03 PM" src="https://github.com/user-attachments/assets/aa8bfa57-4c11-4848-8894-b14366a6a9f9">


## Overview

This repository implements a novel object detection system that leverages diffusion processes and deep learning. The pipeline consists of five main stages: Input, Diffusion Process, Feature Extraction, Processing, and Detection.

## Pipeline Architecture

### 1. Input Stage
* Takes input images in a standardized format
* Prepares images for the diffusion process

### 2. Diffusion Process
The diffusion stage consists of several key components:
* **Original Image**: Initial image input
* **Diffusion Steps**: Sequential application of diffusion transformations
* **Noisy Image t=T**: Final noisy image after diffusion process completion

### 3. Feature Extraction
This stage processes the diffused image through:
* **Timesteps t=0...T**: Multiple temporal sampling points
* **Multi-scale Feature Maps**: Extraction of features at various scales for comprehensive object detection

### 4. Processing
The processing stage includes:
* **UNet Layers**: Deep neural network architecture for feature processing
* **Aggregation Network**: Combines features from different scales
* **Feature Cube**: Consolidated feature representation

### 5. Detection
The final stage performs:
* **Object Detection Head**: Final layer responsible for object detection
* **Output**: Produces bounding boxes for detected objects
* **Training**: Includes original bounding boxes for training purposes

## Key Features
* Multi-scale feature extraction for robust object detection
* Integration of diffusion processes for enhanced feature learning
* UNet-based architecture for detailed feature processing
* Advanced aggregation network for feature combination
* End-to-end trainable pipeline

## Results


### Initial Epochs

![image1](https://github.com/user-attachments/assets/2015a0eb-c3fe-480b-b205-f99d2eb11fcc)

### Intermediate Epochs

![image2](https://github.com/user-attachments/assets/a99f7808-3740-4d94-9e5a-1174944f8406)

### Final Epochs

![image3](https://github.com/user-attachments/assets/3c4005cd-4bd9-4dce-9d81-c718b0decb33)




