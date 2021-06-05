import pandas as pd
import numpy as np
from scipy.io import arff
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import History


class GAN():
    def __init__(self):
        self.noise_dim = 100
        self.row_shape = (9,)
        self.num_examples_to_generate = 16

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
        z = tf.keras.layers.Input(shape=(self.noise_dim,))
        g_rows = self.generator(z)

        self.discriminator.trainable = False
        validity = self.discriminator(g_rows)

        # self.gan =

        self.combined = tf.keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, use_bias=False, input_dim=self.noise_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Dense(32))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(9, activation='tanh'))

        model.summary()

        noise = tf.keras.layers.Input(shape=(self.noise_dim,))
        g_rows = model(noise)
        return tf.keras.Model(noise, g_rows)
        # return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(32, use_bias=False, input_shape=self.row_shape))
        # model.add(layers.Dense(32, use_bias=False, input_shape=(8,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Dense(16))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        # opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.summary()

        g_rows = tf.keras.layers.Input(shape=self.row_shape)
        validity = model(g_rows)
        return tf.keras.Model(g_rows, validity)


    def train(self, df, epochs, batch_size=8):
        # (X_train, _), (_, _) = mnist.load_data()
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)
        X_train = df.values
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        d_losses = np.zeros((epochs, 1))
        d_accuracies = np.zeros((epochs, 1))
        g_losses = np.zeros((epochs, 1))

        for i, epoch in enumerate(range(epochs)):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            rows = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            gen_rows = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(rows, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_rows, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            d_losses[i] = d_loss[0]
            d_accuracies[i] = d_loss[1]
            g_losses[i] = g_loss
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # if epoch % 10 == 0:
            #   noise = np.random.normal(0, 1, (1, self.noise_dim))
            #   gen_rows = self.generator.predict(noise)
            #   print(gen_rows)
        return (d_losses, d_accuracies, g_losses)


    def plot_metric(self, history: History, metric: str = 'loss') -> None:
        import matplotlib.pyplot as plt
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])
        plt.show()