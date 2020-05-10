import tensorflow as tf
import tensorflow.keras as k


def build_generator():
    nfilters = 32
    nchannels = 3
    leaky = 0.2
    kernel_size = 4

    return k.models.Sequential([
        k.layers.Conv2DTranspose(nfilters*16, kernel_size, 1, padding='same', input_shape=(4, 4, 8)),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2DTranspose(nfilters*8, kernel_size, 2, padding='same'),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2DTranspose(nfilters*4, kernel_size, 2, padding='same'),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2DTranspose(nfilters*2, kernel_size, 2, padding='same'),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2DTranspose(nfilters, kernel_size, 1, padding='same'),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2DTranspose(nchannels, kernel_size, 2, padding='same'),
        k.layers.Activation(tf.nn.tanh)
    ])


def build_discriminator():
    nfilters = 4
    leaky = 0.2

    return k.models.Sequential([
        k.layers.Conv2D(nfilters, 3, input_shape=(64, 64, 3)),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.MaxPool2D((2, 2)),
        k.layers.Conv2D(nfilters*2, 3),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.MaxPool2D((2, 2)),
        k.layers.Conv2D(nfilters*4, 3),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.MaxPool2D((2, 2)),
        k.layers.Conv2D(nfilters*8, 3),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(leaky),
        k.layers.Conv2D(nfilters*16, 3),
        k.layers.LeakyReLU(leaky),
        k.layers.Flatten(),
        k.layers.Dense(1, activation='sigmoid')
    ])
