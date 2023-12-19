# Solar Panel Detection

In recent years, the installation of solar panels has increased as renewable energy sources have become more prevalent. Knowing the exact location and footprint of these panels is crucial for energy supply planning, infrastructure optimization, and disaster prediction.

## Project Overview

This project focuses on utilizing Tensorflow and Keras libraries for the segmentation of solar panels in multi-spectral images. The following key tasks are addressed:

- Generating new indices like Normalized Difference Vegetation Index (NDVI), Shadow Index (SI), etc., from multi-spectral images.
- Implementing various architectures such as Unet, Unet++, Unet 3+, etc., to tackle the segmentation challenge.

## Package Usage

Ensure you have the following dependencies installed:

- Python 3.10
- Keras Unet Collection (version 0.1.13)
- NumPy (version 1.23.5)
- TensorFlow (version 2.12.0)

You can install the required packages using the following command:

```bash
pip install python==3.10 keras_unet_collection==0.1.13 numpy==1.23.5 tensorflow==2.12.0
