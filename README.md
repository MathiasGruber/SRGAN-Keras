# SRGAN-Keras
Keras implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)
<img src='images/main_sample.PNG' width="100%" />

## Architecture
The generator creates a high-resolution (HR) image (4x upscaled) from a corresponding low-resolution (LR) image. The discriminator distinguishes the generated (fake) HR images from the original HR images.

<img src='images/architecture.PNG' width="80%" />
__[Figure 4 from paper](https://arxiv.org/abs/1609.04802):__ Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps
(n) and stride (s) indicated for each convolutional layer.

<img src='images/code_setup.PNG' width="80%" />
__Code Overview__: Overview of the three networks; generator, discriminator, and VGG19. Generator create SR image from LR, discriminator predicts whether it's a SR or original HR, and VGG19 extracts features from generated SR and original HR images. 

## Content & Adversarial Loss
<img src='images/loss_equations.PNG' width="80%" />
**Losses Overview**: The perceptual loss is a combination of content loss (based on VGG19 features) and adversarial loss. Equations are taken directly from ["original paper"](https://arxiv.org/abs/1609.04802)
