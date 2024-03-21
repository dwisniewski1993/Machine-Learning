import logging as log

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.python.keras.models import Sequential, Model
from tqdm import tqdm


class GAN:
    def __init__(self) -> None:
        log.getLogger().setLevel(log.INFO)
        log.info('Initialize GAN model...')
        (self.X_train, self.y_train, self.X_test, self.y_test) = self.load_data()

        self.generator = self.define_generator()
        self.discriminator = self.define_discriminator()
        self.model = self.define_gan(generator=self.generator, discriminator=self.discriminator)

    @staticmethod
    def load_data() -> tuple:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

        return x_train, y_train, x_test, y_test

    @staticmethod
    def define_generator() -> Sequential:
        generator = Sequential()

        generator.add(Dense(units=256, input_dim=100))
        generator.add(LeakyReLU(alpha=0.2))

        generator.add(Dense(units=512))
        generator.add(LeakyReLU(alpha=0.2))

        generator.add(Dense(units=1024))
        generator.add(LeakyReLU(alpha=0.2))

        generator.add(Dense(units=784, activation='tanh'))

        generator.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(lr=0.0001, beta_1=0.5))
        generator.summary()

        return generator

    @staticmethod
    def define_discriminator() -> Sequential:
        discriminator = Sequential()
        discriminator.add(Dense(units=1024, input_dim=784))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(rate=0.3))

        discriminator.add(Dense(units=512))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(rate=0.3))

        discriminator.add(Dense(units=256))
        discriminator.add(LeakyReLU(alpha=0.2))

        discriminator.add(Dense(units=1, activation='sigmoid'))

        discriminator.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(lr=0.0001, beta_1=0.5))
        discriminator.summary()

        return discriminator

    @staticmethod
    def define_gan(generator: Sequential, discriminator: Sequential) -> Model:
        discriminator.trainable = False
        gan_input = Input(shape=(100,))
        gen = generator(gan_input)
        gan_output = discriminator(gen)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(lr=0.0001, beta_1=0.5))
        gan.summary()

        return gan

    def train_model(self, epochs: int = 1, batch_size: int = 128) -> None:
        for epoch in range(1, epochs + 1):
            log.info(f"Start epoch {epoch}")
            for _ in tqdm(range(batch_size)):
                noise = np.random.normal(0, 1, [batch_size, 100])

                generated_images = self.generator.predict(noise)
                image_batch = self.X_train[np.random.randint(low=0, high=self.X_train.shape[0], size=batch_size)]
                X = np.concatenate([image_batch, generated_images])

                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_dis)

                noise = np.random.normal(0, 1, [batch_size, 100])
                y_gen = np.ones(batch_size)

                self.discriminator.trainable = False
                self.model.train_on_batch(noise, y_gen)

            if epoch % 50 == 0:
                self.generate_and_plot_results(epoch=epoch, generator=self.generator)

    @staticmethod
    def generate_and_plot_results(epoch: int, generator: Sequential, examples: int = 100, dim: tuple = (10, 10),
                                  fig_size: tuple = (10, 10)) -> None:
        noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100, 28, 28)
        plt.figure(figsize=fig_size)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"gan_generated_image_{epoch}_epoch.png")
