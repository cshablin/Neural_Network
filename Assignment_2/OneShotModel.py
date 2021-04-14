from keras.callbacks import History
from tensorflow.keras import models, optimizers, losses, activations, callbacks, initializers
from tensorflow.keras.layers import *
import tensorflow.keras.backend as backend
import tensorflow as tf


class OneShotModel(object):

    def __init__(self):
        self.__dim = 250
        input_shape = (self.__dim ** 2,)
        convolution_shape = (self.__dim, self.__dim, 1)
        strides = 1
        seq_conv_model = [

            Reshape(input_shape=input_shape, target_shape=convolution_shape),
            Conv2D(32, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            Conv2D(32, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            BatchNormalization(),
            # MaxPooling2D(pool_size=(2, 2), strides=strides),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(64, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            Conv2D(64, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(128, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            Conv2D(128, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.glorot_normal),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(256, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.01)),
            Conv2D(256, kernel_size=(3, 3), strides=1, activation=activations.relu, padding="same", kernel_initializer=initializers.RandomNormal(stddev=0.01)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(1024, activation=activations.relu, kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01)),
            Dense(1024, activation=activations.sigmoid, kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.01))
        ]

        seq_model = tf.keras.Sequential(seq_conv_model)

        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)

        output_x1 = seq_model(input_x1)
        output_x2 = seq_model(input_x2)

        distance_euclid = Lambda(lambda tensors: backend.abs(tensors[0] - tensors[1]))([output_x1, output_x2])
        outputs = Dense(1, activation=activations.sigmoid)(distance_euclid)
        self.__model = models.Model([input_x1, input_x2], outputs)

        self.__model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        print(self.__model.summary())

    def fit(self, x, y, hyper_parameters):
        # This callback will stop the training when there is no improvement in
        # the validation loss for three consecutive epochs.
        callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        # callback = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
        history = self.__model.fit(x, y, batch_size=hyper_parameters['batch_size'], epochs=hyper_parameters['epochs'],
                                   callbacks=[callback], validation_split=hyper_parameters['validation_split'], verbose=2)
        #validation_data=hyper_parameters['val_data'])

        self.__model.summary()
        return history

    def predict(self, x):
        predictions = self.__model.predict(x)
        return predictions

    def evaluate(self, x, y):
        predictions = self.__model.evaluate(x, y)
        return predictions

    # to get a picture of loss progress.
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