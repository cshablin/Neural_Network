from typing import Tuple

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, LeakyReLU, Flatten





class BB_GAN:
    def __init__(self, black_box_model):
        self.latent_dim = 10
        self.generator_vector_size = 8
        self.discriminator_input_shape = (self.generator_vector_size,)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.gan = self.combime_gan(self.generator, self.discriminator)
        # self.gs_rf = joblib.load("random_forest_diab.joblib")
        self.gs_rf = black_box_model


    def combime_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        g_input = Input(shape=(self.latent_dim,))
        c_input = Input(shape=(1,))
        g_sample = g_model([g_input, c_input])
        d_desision = d_model([g_sample, c_input, c_input])

        gan_model = Model([g_input, c_input], d_desision)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return gan_model

    def make_generator_model(self):
        input_noise = Input(shape=(self.latent_dim,))
        c_input = Input(shape=(1,))
        concat_layer_c = concatenate([input_noise, c_input])
        dense_1 = Dense(30, use_bias=True)(concat_layer_c)
        dense_1_bn = BatchNormalization()(dense_1)
        dense_1_lr = LeakyReLU()(dense_1_bn)
        dense_1_do = Dropout(0.2)(dense_1_lr)
        dense_2 = Dense(15)(dense_1_do)
        dense_2_bn = BatchNormalization()(dense_2)
        dense_2_lr = LeakyReLU()(dense_2_bn)
        dense_2_do = Dropout(0.2)(dense_2_lr)
        final_layer = Dense(self.generator_vector_size, activation='tanh')(dense_2_do)
        # concat_layer = concatenate([final_layer, c_input])
        model = Model(inputs=[input_noise, c_input], outputs=final_layer)

        model.summary()
        return model

    def make_discriminator_model(self):

        # input_sample = Input(shape=self.discriminator_input_shape,)
        input_sample = Input(shape=self.discriminator_input_shape)
        cy1_input = Input(shape=(1,))
        cy2_input = Input(shape=(1,))

        concat_layer_c = concatenate([input_sample, cy1_input, cy2_input])
        dense_1 = Dense(32, use_bias=True)(concat_layer_c)
        dense_1_lr = LeakyReLU()(dense_1)
        dense_1_do = Dropout(0.2)(dense_1_lr)
        dense_2 = Dense(16)(dense_1_do)
        dense_2_lr = LeakyReLU()(dense_2)
        dense_2_do = Dropout(0.2)(dense_2_lr)
        flatten = Flatten()(dense_2_do)
        final_layer = Dense(1, activation='sigmoid')(flatten)
        model = Model(inputs=[input_sample, cy1_input, cy2_input], outputs=final_layer)
        opt = tf.keras.optimizers.Adam(lr=0.00015, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def generate_real_x_y(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # prepare samples
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        c_1 = np.random.random((n_samples, ))
        samples_1 = self.generator.predict([noise, c_1])
        y_1 = self.gs_rf.predict_proba(samples_1)[:, 0]
        valid = np.ones((n_samples, 1))
        return samples_1, c_1, y_1, valid

    def generate_fake_x_y(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        c_2 = np.random.random((n_samples, ))
        samples_2 = self.generator.predict([noise, c_2])
        y_2 = self.gs_rf.predict_proba(samples_2)[:, 0]
        invalid = np.zeros((n_samples, 1))
        return samples_2, y_2, c_2, invalid

    def train(self, epochs, batch_size=8):
        # (X_train, _), (_, _) = mnist.load_data()
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)
        # X_train = df.values
        valid = np.ones((batch_size, 1))
        invalid = np.zeros((batch_size, 1))
        valid_twice = np.ones((batch_size * 2, 1))
        # fake = np.zeros((batch_size, 1))
        d_losses = np.zeros((epochs, 1))
        d_accuracies = np.zeros((epochs, 1))
        d_fake_losses = np.zeros((epochs, 1))
        d_fake_accuracies = np.zeros((epochs, 1))
        d_real_losses = np.zeros((epochs, 1))
        d_real_accuracies = np.zeros((epochs, 1))
        g_losses = np.zeros((epochs, 1))
        g_accuracies = np.zeros((epochs, 1))
        best_epoch = 0

        d_acc_delta_to_0_5 = 1
        losses_delta = 1
        real_2_fake_acc_delta = 1
        for i, epoch in enumerate(range(epochs)):
            # prepare samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            noise_ = np.random.uniform(-1, 1, size=(batch_size, self.latent_dim))
            c_1 = np.random.random((batch_size, ))
            samples_1 = self.generator.predict([noise, c_1])
            y_1 = self.gs_rf.predict_proba(samples_1)[:, 0]  # evaluate Y by running RF classifier
            c_2 = np.random.random((batch_size, ))
            samples_2 = self.generator.predict([noise, c_2])
            y_2 = self.gs_rf.predict_proba(samples_2)[:, 0]  # evaluate Y by running RF classifier

            # create training set for the discriminator
            # swapping places between 'y' and 'c' trains the Discriminator
            # to realize that right hand side parameter is fake (and the opposite)
            samples, c, y_bb = np.vstack((samples_1, samples_2)), np.concatenate((c_1, y_2), axis=None), np.concatenate((y_1, c_2), axis=None)
            # samples, c, y_bb = np.vstack((samples_1, samples_2)), np.vstack((c_1, y_2)), np.vstack((y_1, c_2))

            # update discriminator model weights
            d_loss, d_acc = self.discriminator.train_on_batch([samples, c, y_bb], np.vstack((valid, invalid)))

            # evaluate discriminator on real examples
            samples_real, c_real, y_bb_real, y_real = self.generate_real_x_y(batch_size)
            d_loss_real, d_acc_real = self.discriminator.evaluate([samples_real, c_real, y_bb_real], y_real, verbose=0)
            # evaluate discriminator on fake examples
            samples_fake, c_fake, y_bb_fake, y_fake = self.generate_fake_x_y(batch_size)
            d_loss_fake, d_acc_fake = self.discriminator.evaluate([samples_fake, c_fake, y_bb_fake], y_fake, verbose=0)

            d_fake_losses[i] = d_loss_fake
            d_real_losses[i] = d_loss_real
            d_fake_accuracies[i] = d_acc_fake
            d_real_accuracies[i] = d_acc_real

            # create inverted labels for the fake samples so generator can improve to be 'real'
            # update the generator via the discriminator's error
            noise = np.random.normal(0, 1, (batch_size * 2, self.latent_dim))
            c = np.random.normal(0, 1, (batch_size * 2, ))
            g_loss, g_acc = self.gan.train_on_batch([noise, c], valid_twice)

            d_losses[i] = d_loss
            d_accuracies[i] = d_acc
            g_losses[i] = g_loss
            g_accuracies[i] = g_acc



            if i % 50 == 0:
                print("epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss, 100 * d_acc, g_loss, g_acc))
                if 800 <= i:
                    if abs(g_loss - d_loss) < losses_delta: # check that losses converged
                        if d_acc_delta_to_0_5 > abs(d_acc - 0.5) and real_2_fake_acc_delta > abs( d_acc_fake - d_acc_real): # save generator model for closest to 0.5 accuracy
                            losses_delta = abs(g_loss - d_loss)
                            d_acc_delta_to_0_5 = abs(d_acc - 0.5)
                            real_2_fake_acc_delta = abs(d_fake_accuracies - d_real_accuracies)
                            filename = 'generator_model.h5'
                            self.generator.save(filename)
                            best_epoch = epoch

        print(f'Best epoch: {best_epoch}')
        return d_losses, d_accuracies, g_losses, g_accuracies, d_fake_losses, d_real_losses, d_fake_accuracies, d_real_accuracies


class BB_GAN_Cred:
    def __init__(self, black_box_model, latent_dim, gen_out_size, gen_final_activation='tanh'):
        self.latent_dim = latent_dim
        self.generator_vector_size = gen_out_size
        self.discriminator_input_shape = (self.generator_vector_size,)

        self.generator = self.make_generator_model(gen_final_activation)
        self.discriminator = self.make_discriminator_model()
        self.gan = self.combime_gan(self.generator, self.discriminator)
        self.gs_rf = black_box_model

    def combime_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False

        g_input = Input(shape=(self.latent_dim,))
        c_input = Input(shape=(1,))
        g_sample = g_model([g_input, c_input])
        d_desision = d_model([g_sample, c_input, c_input])
        gan_model = Model([g_input, c_input], d_desision)
        # opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.6, beta_2=0.999)
        gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return gan_model

    def make_generator_model(self, gen_final_activation):
        input_noise = Input(shape=(self.latent_dim,))
        c_input = Input(shape=(1,))
        concat_layer_c = concatenate([input_noise, c_input])
        dense_1 = Dense(128, use_bias=True)(concat_layer_c)
        dense_1_bn = BatchNormalization()(dense_1)
        dense_1_lr = LeakyReLU()(dense_1_bn)
        dense_1_do = Dropout(0.25)(dense_1_lr)
        dense_2 = Dense(256)(dense_1_do)
        dense_2_bn = BatchNormalization()(dense_2)
        dense_2_lr = LeakyReLU()(dense_2_bn)
        dense_2_do = Dropout(0.25)(dense_2_lr)
        dense_3 = Dense(512)(dense_2_do)
        dense_3_bn = BatchNormalization()(dense_3)
        dense_3_lr = LeakyReLU()(dense_3_bn)
        dense_3_do = Dropout(0.25)(dense_3_lr)
        final_layer = Dense(self.generator_vector_size, activation=gen_final_activation)(dense_3_do)
        # concat_layer = concatenate([final_layer, c_input])
        model = Model(inputs=[input_noise, c_input], outputs=final_layer)

        model.summary()
        return model


    def make_discriminator_model(self):

        input_sample = Input(shape=self.discriminator_input_shape,)
        cy1_input = Input(shape=(1,))
        cy2_input = Input(shape=(1,))

        concat_layer_c = concatenate([input_sample, cy1_input, cy2_input])
        dense_1 = Dense(512, use_bias=True)(concat_layer_c)
        dense_1_lr = LeakyReLU()(dense_1)
        dense_1_do = Dropout(0.2)(dense_1_lr)
        dense_2 = Dense(256)(dense_1_do)
        dense_2_lr = LeakyReLU()(dense_2)
        dense_2_do = Dropout(0.2)(dense_2_lr)
        dense_3 = Dense(128)(dense_2_do)
        dense_3_lr = LeakyReLU()(dense_3)
        dense_3_do = Dropout(0.2)(dense_3_lr)
        flatten = Flatten()(dense_3_do)
        # layer_final_sm = Dense(2, activation='softmax')(flatten)
        final_layer = Dense(1, activation='sigmoid')(flatten)
        # concat_layer = concatenate([dense_2_do, c_input, y_input])
        model = Model(inputs=[input_sample, cy1_input, cy2_input], outputs=final_layer)

        # opt = tf.keras.optimizers.Adam(lr=0.00015, beta_1=0.5)
        opt = tf.keras.optimizers.Adam(lr=0.000015, beta_1=0.95)  # lr=0.0002, beta_1=0.5, beta_2=0.999
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def generate_real_x_y(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # prepare samples
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        c_1 = np.random.random((n_samples, ))
        samples_1 = self.generator.predict([noise, c_1])
        y_1 = self.gs_rf.predict_proba(samples_1)[:, 0]
        valid = np.ones((n_samples, 1))
        return samples_1, c_1, y_1, valid

    def generate_fake_x_y(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        c_2 = np.random.random((n_samples, ))
        samples_2 = self.generator.predict([noise, c_2])
        y_2 = self.gs_rf.predict_proba(samples_2)[:, 0]
        invalid = np.zeros((n_samples, 1))
        return samples_2, y_2, c_2, invalid

    def train(self, epochs, batch_size=8):
        valid = np.ones((batch_size, 1))
        invalid = np.zeros((batch_size, 1))
        valid_twice = np.ones((batch_size * 2, 1))
        # fake = np.zeros((batch_size, 1))
        d_losses = np.zeros((epochs, 1))
        d_accuracies = np.zeros((epochs, 1))
        d_fake_losses = np.zeros((epochs, 1))
        d_fake_accuracies = np.zeros((epochs, 1))
        d_real_losses = np.zeros((epochs, 1))
        d_real_accuracies = np.zeros((epochs, 1))
        g_losses = np.zeros((epochs, 1))
        g_accuracies = np.zeros((epochs, 1))
        best_epoch = 0
        d_acc_delta_to_0_5 = 1
        losses_delta = 1
        real_2_fake_acc_delta = 1
        for i, epoch in enumerate(range(epochs)):
            # prepare samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            c_1 = np.random.random((batch_size, ))
            samples_1 = self.generator.predict([noise, c_1])
            y_1 = self.gs_rf.predict_proba(samples_1)[:, 0]  # evaluate Y via RF classifier take first column confidence
            c_2 = np.random.random((batch_size, ))
            samples_2 = self.generator.predict([noise, c_2])
            y_2 = self.gs_rf.predict_proba(samples_2)[:, 0]  # evaluate Y via RF classifier take first column confidence

            # create training set for the discriminator
            # swapping places between 'y' and 'c' trains the Discriminator
            # to realize that right hand side parameter is fake (and the opposite)
            samples, c, y_bb = np.vstack((samples_1, samples_2)), np.concatenate((c_1, y_2), axis=None), np.concatenate((y_1, c_2), axis=None)

            # update discriminator model weights
            d_loss, d_acc = self.discriminator.train_on_batch([samples, c, y_bb], np.vstack((valid, invalid)))

            # evaluate discriminator on real examples
            samples_real, c_real, y_bb_real, y_real = self.generate_real_x_y(batch_size)
            d_loss_real, d_acc_real = self.discriminator.evaluate([samples_real, c_real, y_bb_real], y_real, verbose=0)
            # evaluate discriminator on fake examples
            samples_fake, c_fake, y_bb_fake, y_fake = self.generate_fake_x_y(batch_size)
            d_loss_fake, d_acc_fake = self.discriminator.evaluate([samples_fake, c_fake, y_bb_fake], y_fake, verbose=0)

            d_fake_losses[i] = d_loss_fake
            d_real_losses[i] = d_loss_real
            d_fake_accuracies[i] = d_acc_fake
            d_real_accuracies[i] = d_acc_real

            # noise = np.random.normal(0, 1, (batch_size * 2, self.latent_dim)) # Carmel,  WHY TWISE?
            # create inverted labels for the fake samples so generator can improve to be 'real'
            # update the generator via the discriminator's error
            noise = np.random.normal(0, 1, (batch_size * 2, self.latent_dim))
            c = np.random.normal(0, 1, (batch_size * 2, ))
            g_loss, g_acc = self.gan.train_on_batch([noise, c], valid_twice)

            d_losses[i] = d_loss
            d_accuracies[i] = d_acc
            g_losses[i] = g_loss
            g_accuracies[i] = g_acc

            if i % 50 == 0:
                print("epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss, 100 * d_acc, g_loss, g_acc))
                if 800 <= i:
                    if abs(g_loss - d_loss) < 0.1: # check that losses converged
                        if abs(d_acc - 0.5) < 0.05 and abs( d_acc_fake - d_acc_real) < 0.1 : # save generator model for closest to 0.5 accuracy
                            # filename = 'generator_model.h5'
                            filename = 'generator_model_%03d.h5' % epoch

                            self.generator.save(filename)
                            best_epoch = epoch

        print(f'Best epoch: {best_epoch}')
        return d_losses, d_accuracies, g_losses, g_accuracies, d_fake_losses, d_real_losses, d_fake_accuracies, d_real_accuracies