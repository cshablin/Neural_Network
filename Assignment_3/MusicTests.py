import unittest
from typing import List

import numpy as np
import gensim.downloader
from gensim.models import KeyedVectors
from keras.layers import Embedding, LSTM, Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, Sequential
from keras import utils

from Assignment_3.DataLoad import load_lyrics
from Assignment_3.MusicalRNN import LyricsGenerator


class MusicTestCase(unittest.TestCase):

    def test_load_lyrics(self):
        result = load_lyrics("Lyrics\\lyrics_train_set.csv")
        self.assertIsNotNone(result)

    def test_word_embedding(self):
        songs = self.__load_songs("Lyrics\\lyrics_train_set.csv")
        model = KeyedVectors.load_word2vec_format('C:\\Users\\cshablin\\Downloads\\GoogleNews-vectors-negative300.bin', binary=True)

        # Data preprocessing
        tokenize = Tokenizer()
        tokenize.fit_on_texts(songs)
        total_words = len(tokenize.word_index) + 1

        input_sequences = []
        for line in songs:
            token_list = tokenize.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        print(input_sequences[:10])

        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        x_train = input_sequences[:, :-1]
        y_train = input_sequences[:, -1]
        y_train = utils.to_categorical(y_train, num_classes=total_words)

        embedding_dim = 300
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((total_words, embedding_dim))
        for word, i in tokenize.word_index.items():
            # TODO: the model is missing word 'to'
            embedding_vector = model[word]
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        embedding_layer = Embedding(
            total_words,
            embedding_dim,
            embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=False,
        )

        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(units=embedding_dim))
        model.add(Dense(units=total_words))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def __load_songs(self, path) -> List[str]:
        df = load_lyrics(path)
        songs = []
        for song in list(df['lyrics']):
            songs.append(song.replace("&", "silencio").lower())

        return songs


if __name__ == '__main__':
    unittest.main()
