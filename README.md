# Pix2Pix Image-to-Image Translation

A deep learning project using the Pix2Pix conditional GAN model to perform image-to-image translation on aerial images.

## Overview
This project applies the Pix2Pix model to the TU-Graz dataset, which consists of 400 aerial images captured by a drone. The goal is to generate realistic images from label maps, optimizing performance through data augmentation and loss function modifications.

## Features
- Implements the **Pix2Pix GAN** model.
- Uses the **TU-Graz aerial image dataset**.
- **Baseline implementation** based on existing Pix2Pix repositories.
- **Enhancements:**
  1. **Data Augmentation** (rotation, cropping, flipping).
  2. **Modified Loss Function** (Kullback–Leibler divergence integration).
- **Performance Metrics:** VIF, UQI, SSIM, and PSNR.


## Performance Comparison
Below is a comparison of different configurations tested:

| Model      | VIF       | UQI       | SSIM      | PSNR (dB)  |
| ---------- | --------- | --------- | --------- | ---------- |
| Baseline   | 0.126     | 0.039     | 0.187     | 13.400     |
| Base + KLD | 0.113     | 0.045     | 0.226     | 13.643     |
| Augmented  | 0.153     | 0.044     | 0.236     | 13.808     |
| Aug + KLD  | **0.183** | **0.053** | **0.242** | **13.894** |


## For more information there is a report that explains the project in detail

## References
- [📄 Pix2Pix Paper (Isola et al., 2018)](https://arxiv.org/abs/1611.07004)
- [📂 TU-Graz Dataset](https://www.tugraz.at/institute/icg/research/team-fraundorfer/software-media/dronedataset)
- [💾 Original Pix2Pix Repository](https://github.com/phillipi/pix2pix)

---
👥 **Contributors:** *Stefano Iannicelli & Ettore Caputo*
