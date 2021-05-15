import numpy as np
from keras import callbacks
from keras.callbacks import History
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model


class LyricsGenerator(object):

    def __init__(self, embedding_dim: int, vocab_size: int, input_size: int, embedding_matrix: np.ndarray):
        embedding_layer = Embedding(
            vocab_size,
            embedding_dim,
            input_length=input_size,
            weights=[embedding_matrix],
            # embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=False,
        )

        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(LSTM(units=embedding_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=vocab_size, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def fit(self, x, y, hyper_parameters):
        callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        callback2 = callbacks.LearningRateScheduler(self._lr_scheduler)
        callback3 = callbacks.ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
        # callback = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
        # history = self.model.fit(x, y, epochs=100, verbose=1)
        history = self.model.fit(x, y, batch_size=hyper_parameters['batch_size'], epochs=hyper_parameters['epochs'],
                                 callbacks=[callback, callback2, callback3],
                                 verbose=2, validation_split=hyper_parameters['validation_split'],
                                 validation_data=None)
        self.model.summary()
        self.model.save('lyrics_model.h5')
        return history

    def _lr_scheduler(self, epoch, lr):
        return 0.99 * lr

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


class LyricsEarlyFusionGenerator(object):

    def __init__(self, embedding_dim: int, vocab_size: int, sequence_length: int, midi_features_size: int, embedding_matrix: np.ndarray):

        input_lyrics = Input(shape=(sequence_length,))
        embedding_layer = Embedding(
            vocab_size,
            embedding_dim,
            input_length=sequence_length,
            weights=[embedding_matrix],
            # embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=True,
        )(input_lyrics)

        # input_midi = Input(shape=(midi_features_size,))
        input_midi = Input(shape=(sequence_length, midi_features_size,))
        embedding_midi_vec = concatenate([embedding_layer, input_midi])
        lstm = LSTM(units=embedding_dim + midi_features_size)(embedding_midi_vec)
        dropout = Dropout(0.2)(lstm)
        final_model = Dense(units=vocab_size, activation='softmax')(dropout)
        self.model = Model(inputs=[input_lyrics, input_midi], outputs=final_model)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def fit(self, x, y, hyper_parameters):
        callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        callback2 = callbacks.LearningRateScheduler(self._lr_scheduler)
        # callback3 = callbacks.ModelCheckpoint('lyrics_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        # callback = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
        # history = self.model.fit(x, y, epochs=100, verbose=1)
        history = self.model.fit(x, y, batch_size=hyper_parameters['batch_size'], epochs=hyper_parameters['epochs'],
                                 callbacks=[callback],
                                 # callbacks=[callback, callback2],
                                 verbose=1, validation_split=hyper_parameters['validation_split'],
                                 validation_data=hyper_parameters['val_data'])
        self.model.save('lyrics_model.h5')

        return history

    def _lr_scheduler(self, epoch, lr):
        return 0.95 * lr

    def evaluate(self, x, y):
        predictions = self.model.evaluate(x, y)
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