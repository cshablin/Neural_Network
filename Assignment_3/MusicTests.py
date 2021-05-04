import string
import unittest
from typing import List, Tuple
import re

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

        # Data preprocessing
        tokenize = Tokenizer()
        tokenize.fit_on_texts(songs)
        total_words = len(tokenize.word_index) + 1

        embedding_dim = 300
        embedding_matrix = self.prepare_embeddings(embedding_dim, tokenize, total_words)

        x_train, y_train = self.create_x_y_train(songs, tokenize, total_words)

        parameters = {
            'batch_size' : 8 ,
            'validation_split' : 0.2 ,
            'epochs' : 15 ,
            'val_data' : None
        }
        lyrics_generator = LyricsGenerator(embedding_dim, total_words, x_train.shape[1], embedding_matrix)
        h = lyrics_generator.fit(x_train, y_train, parameters)
        lyrics_generator.plot_metric(h)

    def create_x_y_train(self, songs, tokenize, total_words) -> Tuple[np.ndarray, np.ndarray]:
        input_sequences = []
        for line in songs:
            token_list = tokenize.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        print(input_sequences[:10])
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        x_train = input_sequences[:, :-1]
        y_train = input_sequences[:, -1]
        y_train = utils.to_categorical(y_train, num_classes=total_words)
        return x_train, y_train

    def prepare_embeddings(self, embedding_dim, tokenize, total_words):
        # Prepare embedding matrix

        embedding_matrix = np.load("Lyrics\\embedding_matrix.npy")
        # hits = 0
        # misses = 0
        # model = KeyedVectors.load_word2vec_format('C:\\Users\\cshablin\\Downloads\\GoogleNews-vectors-negative300.bin',
        #                                           binary=True)
        # non_existing_words_distribution_mu = -0.9
        # missing_words = 1000
        # delta_mu = 0.9 * 2 / missing_words
        # embedding_matrix = np.zeros((total_words, embedding_dim))
        # for word, i in tokenize.word_index.items():
        #     try:
        #         embedding_vector = model[word]
        #         embedding_matrix[i] = embedding_vector
        #         hits += 1
        #     except KeyError as e:
        #         # Words not found in embedding index will be sampled from N(mu,0.01)
        #         embedding_matrix[i] = np.random.normal(non_existing_words_distribution_mu, delta_mu / 10, embedding_dim)
        #         non_existing_words_distribution_mu += delta_mu
        #         misses += 1
        # print("Converted %d words (%d misses)" % (hits, misses))
        # np.save("Lyrics\\embedding_matrix.npy", embedding_matrix)
        return embedding_matrix

    def __load_songs(self, path) -> List[str]:
        df = load_lyrics(path)
        songs = []
        for song in list(df['lyrics']):
            modified_song = re.sub(r"\([^()]*\)", "", song)
            # modified_song = re.sub(r"\[[^()]*\]", "", modified_song)
            # for ch in ["[", "]", "chorus"]:
            #     modified_song = modified_song.replace(ch, "")
            modified_song = modified_song.replace("chorus", "").replace("&", "silencio").lower()
            # modified_song = modified_song.replace("&", "silencio").lower()
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            modified_song = regex.sub('', modified_song)
            songs.append(modified_song)

        return songs


if __name__ == '__main__':
    unittest.main()
