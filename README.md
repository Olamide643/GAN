
# GAN Model with Generator and Discriminator

This repository contains an implementation of a Generative Adversarial Network (GAN) using TensorFlow/Keras. The GAN consists of a generator and discriminator model that work together to generate realistic images.

## Overview

The GAN is designed to learn the distribution of real images and generate new, realistic images. The generator creates new images, while the discriminator evaluates them as real or fake. Both models are trained in a competitive process where the generator tries to trick the discriminator, and the discriminator tries to improve its accuracy in distinguishing real from fake images.

## Features

- **Generator Model:**
  - Upsampling and Convolutional layers to generate images from random noise.
  - Leaky ReLU activations to introduce non-linearity.
  - Sigmoid activation to output grayscale images.

- **Discriminator Model:**
  - Convolutional layers to extract features from images.
  - Leaky ReLU activations for non-linearity.
  - Dropout for regularization to avoid overfitting.
  - Flatten and Dense layers to classify images as real or fake.

- **GAN Training:**
  - The GAN model class inherits from `tf.keras.Model`.
  - Custom `train_step` to manage the training of both generator and discriminator.
  - Gradient Tape used for backpropagation and optimization.

## Installation

To run this project, you will need:

- Python 3.x
- TensorFlow 2.x

You can install the dependencies using:

```bash
pip install tensorflow
```

## Usage

1. **Generator and Discriminator Models:**
   The `build_generator` function defines the generator model, which transforms random noise into images. The `build_discriminator` function defines the discriminator model, which classifies images as real or fake.

2. **Training the GAN:**
   The `GAN` class manages the training of both models. The generator tries to generate images that fool the discriminator, while the discriminator tries to distinguish between real and fake images.

3. **Example Usage:**

```python
# Initialize the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile the GAN
gan = GAN(generator, discriminator)
gan.compile(gen_opt=tf.keras.optimizers.Adam(), dis_opt=tf.keras.optimizers.Adam(),
            gen_loss=tf.keras.losses.BinaryCrossentropy(), dis_loss=tf.keras.losses.BinaryCrossentropy())

# Train the GAN (example batch input is required)
gan.train_step(batch_of_real_images)
```

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

