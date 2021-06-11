from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import History


class GAN:
    def __init__(self, latent_dim, gen_out_size, gen_dense_sizes: List[int], dis_dense_sizes: List[int], gen_final_activation='tanh' ):
        self.latent_dim = latent_dim
        self.generator_vector_size = gen_out_size
        self.discriminator_input_shape = (self.generator_vector_size,)

        self.generator = self.make_generator_model(gen_dense_sizes, gen_final_activation)
        self.discriminator = self.make_discriminator_model(dis_dense_sizes)
        self.gan = self.combime_gan(self.generator, self.discriminator)

    def combime_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect generator and discriminator
        model = tf.keras.Sequential()
        model.add(g_model)
        model.add(d_model)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5) # lr=0.0002, beta_1=0.5, beta_2=0.999
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def make_generator_model(self, dis_dense_sizes: List[int], gen_final_activation):
        model = tf.keras.Sequential()

        first_layer = True
        for layer_size in dis_dense_sizes:
            if first_layer:
                model.add(layers.Dense(layer_size, input_dim=self.latent_dim))
                first_layer = False
            else:
                model.add(layers.Dense(layer_size))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.25))

        model.add(layers.Dense(self.generator_vector_size, activation=gen_final_activation))

        model.summary()
        return model

    def make_discriminator_model(self, dis_dense_sizes: List[int]):
        model = tf.keras.Sequential()
        first_layer = True
        for layer_size in dis_dense_sizes:
            if first_layer:
                model.add(layers.Dense(layer_size, input_dim=self.generator_vector_size))
                first_layer = False
            else:
                model.add(layers.Dense(layer_size))
            # model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        # Carmel - note for report, lr made lower here x10 than Generator because
        # when equel the D outpreformed G and loss could not converge so idea is to make D learn slower so that G could be an adversary

        # opt = tf.keras.optimizers.Adam(lr=0.00002, beta_1=0.5)  # lr=0.0002, beta_1=0.5, beta_2=0.999
        # opt = tf.keras.optimizers.Adam(lr=0.00002, beta_1=0.8)  # lr=0.0002, beta_1=0.5, beta_2=0.999
        # opt = tf.keras.optimizers.Adam(lr=0.000015, beta_1=0.95)  # lr=0.0002, beta_1=0.5, beta_2=0.999
        opt = tf.keras.optimizers.Adam(lr=0.000015, beta_1=0.90,beta_2=0.99 )  # lr=0.0002, beta_1=0.5, beta_2=0.999
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model



    def generate_real_x_y(self, data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        y_real = np.ones((n_samples, 1))
        idx = np.random.randint(0, data.shape[0], n_samples)
        x_real = data[idx]
        return x_real, y_real

    def generate_fake_x_y(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        y_fake = np.zeros((n_samples, 1))
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        x_fake = self.generator.predict(noise)
        return x_fake, y_fake


    def train(self, df, epochs, batch_size=8):
        # (X_train, _), (_, _) = mnist.load_data()
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)
        X_train = df.values
        valid = np.ones((batch_size, 1))
        valid_twice = np.ones((batch_size * 2, 1))
        fake = np.zeros((batch_size, 1))
        d_losses = np.zeros((epochs, 1))
        d_accuracies = np.zeros((epochs, 1))
        d_fake_losses = np.zeros((epochs, 1))
        d_fake_accuracies = np.zeros((epochs, 1))
        d_real_losses = np.zeros((epochs, 1))
        d_real_accuracies = np.zeros((epochs, 1))
        g_losses = np.zeros((epochs, 1))
        g_accuracies = np.zeros((epochs, 1))

        for i, epoch in enumerate(range(epochs)):
            # prepare real fake examples
            x_fake, y_fake = self.generate_fake_x_y(batch_size)
            x_real, y_real = self.generate_real_x_y(X_train, batch_size)

            # Carmel - try this
            # set discriminator learning rate:
            # current_learning_rate = self.discriminator.optimizer.learning_rate * 0.999
            # backend.set_value(self.discriminator.optimizer.learning_rate, current_learning_rate)  # set new learning_rate


            # create training set for the discriminator
            x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, d_acc = self.discriminator.train_on_batch(x, y)

            # evaluate discriminator on real examples
            x_real, y_real = self.generate_real_x_y(X_train, batch_size)
            d_loss_real, d_acc_real = self.discriminator.evaluate(x_real, y_real, verbose=0)
            # evaluate discriminator on fake examples
            x_fake, y_fake = self.generate_fake_x_y(batch_size)
            d_loss_fake, d_acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)

            d_fake_losses[i] = d_loss_fake
            d_real_losses[i] = d_loss_real
            d_fake_accuracies[i] = d_acc_fake
            d_real_accuracies[i] = d_acc_real

            noise = np.random.normal(0, 1, (batch_size * 2, self.latent_dim))
            # create inverted labels for the fake samples so generator can improve to be 'real'
            # update the generator via the discriminator's error
            g_loss, g_acc = self.gan.train_on_batch(noise, valid_twice)

            d_losses[i] = d_loss
            d_accuracies[i] = d_acc
            g_losses[i] = g_loss
            g_accuracies[i] = g_acc

            if i % 50 == 0:
                print("epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss, 100 * d_acc, g_loss, g_acc))
                # if 800<=i and i<= 1200:
                #   filename = 'generator_model_%03d.h5' % (epoch + 1)
                #   self.generator.save(filename)

        return d_losses, d_accuracies, g_losses, g_accuracies, d_fake_losses, d_real_losses, d_fake_accuracies, d_real_accuracies

