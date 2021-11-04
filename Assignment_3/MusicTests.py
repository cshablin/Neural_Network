import os
import random
import string
import unittest
from typing import List, Tuple
import re
import json

import numpy as np
# import gensim.downloader
# from gensim.models import KeyedVectors
# from keras.layers import Embedding, LSTM, Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, Sequential
from keras import utils

from Assignment_3.DataLoad import *
from Assignment_3.Midi import load_midi
from Assignment_3.MusicalRNN import LyricsGenerator# , LyricsGeneratorLateFusion


class MusicModelOneTestCase(unittest.TestCase):

    def test_load_lyrics(self):
        result = load_lyrics("Lyrics\\lyrics_train_set.csv")
        self.assertIsNotNone(result)

    def test_word_embedding(self):
        songs = self.__load_songs("Lyrics\\lyrics_train_set.csv")
        test_songs = self.__load_songs("Lyrics\\lyrics_test_set.csv", pop_305=False)


        # Data preprocessing
        tokenize = Tokenizer()
        tokenize.fit_on_texts(songs)
        total_words = len(tokenize.word_index) + 1

        embedding_dim = 300
        cleaned_songs, embedding_matrix = self.prepare_embeddings_2(songs, embedding_dim, tokenize)
        cleaned_test_songs, _ = self.prepare_embeddings_2(test_songs, embedding_dim, tokenize)


        # embedding_matrix = self.prepare_embeddings(embedding_dim, tokenize, total_words)

        x_train, y_train, x_v, y_v = self.create_x_y_train(cleaned_songs, tokenize, total_words)
        x_test, y_test = self.create_x_y(cleaned_test_songs, tokenize, total_words)

        parameters = {
            'batch_size' : 8 ,
            'validation_split' : 0.2 ,
            'epochs' : 15 ,
            'val_data' : None
        }
        lyrics_generator = LyricsGenerator(embedding_dim, total_words, x_train.shape[1], embedding_matrix)
        h = lyrics_generator.fit(x_train, y_train, parameters)
        lyrics_generator.plot_metric(h)

    def prepare_embeddings_2(self, songs, embedding_dim, tokenize):
        # hits = 0
        # misses = 0
        # import zipfile
        # glove_word2vec = {}
        # # with zipfile.ZipFile('C:\\Users\\cshablin\\Downloads\\glove.6B.zip', "r") as zip_ref:
        # #     zip_ref.extractall("temp_dir")
        # with open("temp_dir\\glove.6B.300d.txt", "r", encoding="utf8") as f:
        #     line_index = 0
        #     for line in f:
        #         try:
        #             # Note: use split(' ') instead of split() if you get an error.
        #             values = line.split(' ')
        #             word = values[0]
        #             coefs = np.asarray(values[1:], dtype='float32')
        #             glove_word2vec[word] = coefs
        #             line_index += 1
        #         except Exception as e:
        #             print(e)
        #
        # embedding_matrix = np.zeros((len(tokenize.word_index) + 1, embedding_dim))
        # non_words_list = []
        # for word, i in tokenize.word_index.items():
        #     try:
        #         embedding_vector = glove_word2vec[word]
        #         embedding_matrix[i] = embedding_vector
        #         hits += 1
        #     except KeyError as e:
        #         non_words_list.append(word)
        # print("Converted %d words (%d misses)" % (hits, misses))
        # np.save("Lyrics\\embedding_matrix_glove.npy", embedding_matrix)

        embedding_matrix = np.load("Lyrics\\embedding_matrix_glove.npy")
        non_words_dict = json.load(open("Lyrics\\non_words.json", "r"))
        non_words_list = non_words_dict["non existing words"]

        cleaned_songs = self.remove(non_words_list, songs)

        # non_words = {}
        # non_words["non existing words"] = non_words_list
        # with open("Lyrics\\non_words.json", "w") as fp:
        #     json.dump(non_words, fp, indent=True)
        # os.chmod("Lyrics\\non_words.json", 0o644)
        return cleaned_songs, embedding_matrix

    def create_x_y_train(self, songs, tokenize, total_words) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_sequences = []
        random.shuffle(songs)

        train_songs = songs[:120]
        val_songs = songs[120:]
        # for line in songs:
        #     token_list = tokenize.texts_to_sequences([line])[0]
        #     for i in range(1, len(token_list)):
        #         n_gram_sequence = token_list[:i + 1]
        #         input_sequences.append(n_gram_sequence)
        # print(input_sequences[:10])
        # max_sequence_len = max([len(x) for x in input_sequences])
        # # input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        # input_sequences = np.array(pad_sequences(input_sequences, maxlen=100, padding='pre'))
        # x_train = input_sequences[:, :-1]
        # y_train = input_sequences[:, -1]
        # y_train = utils.to_categorical(y_train, num_classes=total_words)
        x_t, y_t = self.create_x_y(train_songs, tokenize, total_words)
        x_v, y_v = self.create_x_y(val_songs, tokenize, total_words)
        return x_t, y_t, x_v, y_v

    def create_x_y(self, songs, tokenize, total_words) -> Tuple[np.ndarray, np.ndarray]:
        input_sequences = []
        for line in songs:
            token_list = tokenize.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)
        print(input_sequences[:10])
        max_sequence_len = max([len(x) for x in input_sequences])
        # input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=10, padding='pre'))
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

    def __load_songs(self, path, pop_305=True) -> List[str]:
        df = load_lyrics(path)
        songs = []
        for song in list(df['lyrics']):
            song += " EOF"
            # remove '(*)'
            modified_song = re.sub(r"\([^()]*\)", "", song)
            modified_song = modified_song.replace("chorus", "").lower()
            # modified_song = modified_song.replace("chorus", "").replace("&", "silencio").lower()
            # modified_song = modified_song.replace("&", "silencio").lower()
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            modified_song = regex.sub('', modified_song)
            songs.append(modified_song)
        if pop_305:
            songs.pop(305)
        return songs

    def remove(self, words: List[str], songs):
        resulting_songs = []
        for song in songs:
            result = song
            for word in words:
                result = result.replace(word, '')
            resulting_songs.append(result)
        return resulting_songs


class MusicModelTwoTestCase(unittest.TestCase):

    def test_late_fusion_model(self):
        df_songs = load_songs('Lyrics\\lyrics_train_set.csv')
        # df_midi = load_midi("MIDI\\")
        df_midi = load_midi("temp_dir\\midi_files\\")
        df_songs['filename'] = df_songs['artist'].str.replace(' ', "_") + "_-_" + df_songs['title'].str.replace(' ', "_") + ".mid"
        df_merged = pd.merge(df_songs, df_midi, on="filename")
        tokenize = Tokenizer(filters='')
        tokenize.fit_on_texts(df_merged['lyrics'])
        total_words = len(tokenize.word_index) + 1
        embedding_dim = 300
        embedding_matrixs = np.load('Lyrics\\embedding_matrix_6b.npy')


        non_words_dict = json.load(open("Lyrics\\non_words.json", "r"))
        non_words_list = non_words_dict["non existing words"]
        df_merged['lyrics'] = remove_words(non_words_list, df_merged['lyrics'])

        max_sequence_len = 8
        df_train, df_val = split_df_to_train_val(df_merged, ratio=0.9)
        x_train, x_train_midi, y_train = create_x_y(df_train, tokenize, total_words, max_sequence_len)
        x_val, x_val_midi, y_val = create_x_y(df_val, tokenize, total_words, max_sequence_len)
        x_train_midi = np.asarray(x_train_midi).astype('float32')
        x_val_midi = np.asarray(x_val_midi).astype('float32')

        np.broadcast_to()

        # Train model
        parameters = {
            'batch_size' : 128 ,
            'validation_split' : None ,
            'epochs' : 7 ,
            'val_data' :  ([x_val, x_val_midi], y_val)
        }
        lyrics_generator = LyricsGeneratorLateFusion(embedding_dim, total_words, x_train.shape[1], x_train_midi.shape[1], embedding_matrixs)
        # increasing dimension to match Embedding layer output dimension
        x_train_midi = np.repeat(x_train_midi[:, np.newaxis, :], x_train.shape[1], axis=1)

        h = lyrics_generator.fit([x_train, x_train_midi], y_train, parameters)
        lyrics_generator.plot_metric(h)
        lyrics_generator.plot_metric(h, metric="accuracy")

    def test_increase_dimensions_by_replicate(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        print(a.shape)
        # (2,  2)

        # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
        # array along, (you can achieve the same effect by indexing with None, see below)
        # b = np.repeat(a[:, :, np.newaxis], 5, axis=2)
        b = np.repeat(a[:, np.newaxis, :], 5, axis=1)

    def test_duplicate_rows(self):
        a = np.array([1, 2, 3])
        print(a.shape)
        b = np.repeat(a[np.newaxis, :], 5, axis=0)

        # indexing with np.newaxis inserts a new 3rd dimension, which we then repeat the
        # array along, (you can achieve the same effect by indexing with None, see below)
        # b = np.repeat(a[:, :, np.newaxis], 5, axis=2)
        b = np.repeat(a, 5, axis=0)
        print(b.shape)


if __name__ == '__main__':
    unittest.main()
