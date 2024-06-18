# Extreme Low Light Image Denoising  
# What it does ❓
The aim of the project was to denoise images taken in extreme low light conditions. We often find ourselves with these low light images due to technical and environmental constraints. This results in loss of information. While there are many techniques to denoise such images, most of them are computationally expensive. Therefore, we can use Deep neural networks to perform the task. 
# EXAMPLE 
![image](https://github.com/lazy-insomaniac/Extreme-Low-Light-Image-Denoising/assets/114395022/e5badc61-90ca-4bba-9112-ef6c09625917)
# Description 📝
Zero Reference Deep Curve Estimation and Learning to see in the Dark Papers are implemented through the use of DCE Net and UNet. A modified version of Zero DCE is implemented called Partial-DCE.
The main file for my code are in there respective folders you just have to run the file with your images in 'test/low/' and you will get output images in 'test
This study primarily focuses on implementing UNet and an  improved version of Zero DCE.
-	Obtain high quality light enhanced images  
-	Obtain output images with high PSNR ratio

# Note ⚠️
- For the UNet model , you will have to download the model file seperately and store it into UNet folder before running main.py

# Download Links: 🔗
* Unet model weights file: https://drive.google.com/file/d/1Ov7xwG9VEw6F1JLK3MOKL8BC4GuCNCWu/view?usp=drive_link 
* Model Description🤖:
  -Partial DCE
    -  A lightweight deep network, DCE-Net is used to estimate pixel-wise and high-order curves for dynamic range adjustment of a given 
      image. The curve estimation is specially designed, considering pixel value range, monotonicity, and differentiability. Zero DCE does not require any paired 
      data, this is done by set of non-reference loss functions. Since we have access to a dataset that includes both low-light input images and  their corresponding high-quality reference images, we can        
      leverage this information to improve Zero-DCE denoising model.By incorporating these reference loss functions into our training process, our model learned to produce denoised images that closely resemble
      the high quality references. This approach will often lead to higher PSNR scores compared to models trained solely on low-light input images without reference supervision.
    - PSNR Values    TRAIN = 17.39      TEST/VAL = 19.50       PAPER = 16.57     

  -UNet
    -The architecture is called U-Net due to its U-shaped structure, which consists of a contracting path (encoder) and an expansive path (decoder).U-Net’s architecture, with its ability to capture both local and      global features through its contracting and expansive paths, makes it highly suitable for low-light image denoising tasks, providing clear and high-quality images.
    - PSNR Values    TRAIN = 17.31      TEST/VAL = 19.13       PAPER = 28.88
* LOL-DATASET dataset📊 : https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
* Dataset Description:
- The LOL dataset is composed of  low-light and normal-light image pairs. The low-light images contain noise produced during the photo capture process. Most of the images are indoor scenes. All the images have a resolution of 400×600. The dataset was introduced in the paper Deep Retinex Decomposition for Low-Light Enhancement.
  - Statistics 
    - Total number of examples: 500 
    - Image size: 600x400 pixels, color (RGB channels) png
  - Train / Dev (Validation) / Test Splits
  - Training set: 485(final training)  400(inital training)
  - Validation
  - Test set: 15 images

# Installation 🔧
  - Requiremnts:
    - Python 
    - Torch 
    - TorchVision
    - Tensorflow
    - Keras
    - OpenCV (cv2)
    -  cuda : 11.8 (P100 and T4x2 GPU)(for Training)
    -  NumPy: 1.16.4 or higher
    - Optuna
    - VGG 19
    - PIL

# Documentation 📑
 - HLD: Project_Report: [https://drive.google.com/file/d/1Ov7xwG9VEw6F1JLK3MOKL8BC4GuCNCWu/view?usp=drive_link](https://drive.google.com/file/d/1Bg57Z38PaLd2TcANpoFvaHQpaugCrC6P/view?usp=drive_link)

# References ⚓
 - Zero Reference Deep Curve Estimation for Low Light Enhancement : https://arxiv.org/pdf/2001.06826
 - Learning to see in the Dark :  https://arxiv.org/pdf/1805.01934 
