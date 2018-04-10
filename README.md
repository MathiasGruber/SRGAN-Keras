# SRGAN-Keras
Keras implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

## Architecture
The generator creates a high-resolution (HR) image (4x upscaled) from a corresponding low-resolution (LR) image. The discriminator distinguishes the generated (fake) HR images from the original HR images.

<img src='images/architecture.PNG' width="80%" />
[**Figure 4**](https://arxiv.org/abs/1609.04802): Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps
(n) and stride (s) indicated for each convolutional layer.

## Content & Adversarial Loss
TODO: Details on loss implementation