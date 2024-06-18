# Extreme Low Light Image Denoising  
# What it does ❓
The aim of the project was to denoise images taken in extreme low light conditions. We often find ourselves with these low light images due to technical and environmental constraints. This results in loss of information. While there are many techniques to denoise such images, most of them are computationally expensive. Therefore, we can use Deep neural networks to perform the task.                                                                                                                             

# Description 📝
Zero Reference Deep Curve Estimation and Seeing in the Dark Papers are implemented through the use of DCE Net and UNet. A modified version of Zero DCE is implemented called Partial-DCE.
The main file for my code are in there respective folders , for UNet you will need to to download the weights file seperartely link for which has been provided below
This study primarily focuses on implementing UNet and an  improved version of Zero DCE.
-	Obtain high quality light enhanced images  
-	Obtain output images with high PSNR ratio

# Note 
- The Partial DCE model 

# Download Links: 🔗
* Pretrained DeiT model from hugging face: https://huggingface.co/facebook/deit-base-patch16-224 
* Model Description🤖:


* LOL-DATASET dataset📊 : https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
* Dataset Description:

  - Statistics 
    - Total number of examples: 500 
    - Image size: 600x400 pixels, color (RGB channels) png
  - Train / Dev (Validation) / Test Splits
  - Training set: 50,000 images
  - Test set: 10,000 images
 - Note: The CIFAR-10 dataset does not come with a predefined development (validation) set. Researchers often partition the training set to create a validation set for hyperparameter tuning and model evaluation during development. A common practice is to use 45,000 images for training and set aside 5,000 for validation.

- The Python codes provided in this repository download and implement the model and the dataset in the code itself, as they are directly callable. There is no requirement of setting paths in the code or placing files in specifc folders.
# Installation 🔧
  - Requiremnts:
    - Python : 3.11
    - Torch : 2.2.0
    - TQDM : 4.66.1
    - TorchVision:
    - HuggingFace library of Transformers : 4.17.0
    -  cuda : 11.8 (P100 GPU Accelerator)
    -  NumPy: 1.16.4 or higher

# Setup ⚙️

<img width="904" alt="Screenshot 2024-02-10 at 1 28 33 AM" src="https://github.com/Swadesh06/BYOP_Repro_UPop/assets/129365476/12635e40-da77-48b9-acd6-8d899511fc3d">

# Documentation 📑
 - HLD: BYOP_ReproducibilityTrack_2024_Report: https://docs.google.com/document/d/1RguUkhHiGCgGGspkQwgJ_yfFekhuNIvs/edit
   
