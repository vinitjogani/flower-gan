import numpy as np
from load import load_from_dir, view
import tensorflow as tf
import tensorflow.keras as k
from model import build_discriminator, build_generator
import matplotlib.pyplot as plt

X = load_from_dir("flowers")
discriminator = build_discriminator()
generator = build_generator()
d_optimizer = k.optimizers.RMSprop()
g_optimizer = k.optimizers.RMSprop()


def generate_true_samples(batch_size=64):
    b = np.random.randint(0, len(X), size=(batch_size,))
    return X[b], np.ones((batch_size, 1))


def generate_fake_samples(batch_size=64):
    z = np.random.randn(batch_size, 4, 4, 8)
    return generator(z), np.zeros((batch_size, 1))


def train_discriminator(B=64):
    X1, y1 = generate_true_samples(B)
    X2, y2 = generate_fake_samples(B)
    X, y = np.vstack([X1, X2]), np.vstack([y1, y2])
    noise = np.random.rand(len(y), 1)
    y = y + 0.05 * noise

    with tf.GradientTape() as tape:
        pred = discriminator(X)
        loss = tf.reduce_mean(k.losses.binary_crossentropy(y, pred))
    grads = tape.gradient(loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return loss.numpy()


def train_generator(B=64):

    with tf.GradientTape() as tape:
        X, y = generate_fake_samples(B)
        pred = discriminator(X)
        loss = tf.reduce_mean(k.losses.binary_crossentropy(1-y, pred))
    grads = tape.gradient(loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return loss.numpy()


if __name__ == '__main__':

    for step in range(500):
        disc_loss = train_discriminator()
        gen_loss = train_generator()

        if (step+1) % 5 == 0:

            X_fake = generate_fake_samples(4)[0]
            X_fake = X_fake.numpy().reshape((-1, 64, 64, 3))

            fig, ax = plt.subplots(1, 4, figsize=(8, 2))
            for i in range(len(ax)):
                view(X_fake[i], ax[i])
            plt.show()

            print(f"Epoch {step+1} of 100", end='\t\t')
            print(f"Disc Loss={disc_loss}", end='\t\t')
            print(f"Gen Loss={gen_loss}")
