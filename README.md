# Aircraft recognition and generation

## 1. Methodology

### 1.1 Multi-class Classification

For the Convolutional Neural Network project (CNN), we used the  Military Aircraft Detection dataset  as specified. Then, for the code itself, we import all the necessary libraries, such as TensorFlow and Keras which contained the models and layers, necessary to train the CNN. We also imported the libraries which allowed us to check the file extension and the treatment of the pictures into an array (NumPy)

For the deep learning model, 4 layers were added so that each layer is composed of:
- One convolutional layer, for extracting features from the images in our data set.
- One pooling layer, helping with the overfitting problem. In other words, we selected the maximum for a set of regions.

For the final layer, we use a flattening layer in order to have a single dimension, after a single dense layer is used to have a single output.

Finally, we compile the model using the predefined compile method offered by the Keras library to configure the model for training.

### 1.2 Image Generation

In the generator part, we use Deep Convolutional Generative Adversarial Network, which is a direct extension of general Generative Adversarial Network.

Architecture guidelines for stable Deep Convolutional GANs:
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.
