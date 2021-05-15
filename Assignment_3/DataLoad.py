import re
import string
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.preprocessing.sequence import pad_sequences
from keras import utils


def load_lyrics(csv_path: str) -> DataFrame:
    df = pd.read_csv(csv_path, sep='\n', header=None)
    res = df.iloc[:, 0].str.rstrip(r'&, ').str.extract(r'([^,]+),([^,]+),(.+)')
    res.columns = ['artist', 'title', 'lyrics']
    return res


def load_songs(path) -> DataFrame:
    df = load_lyrics(path)
    for i, song in enumerate(list(df['lyrics'])):

        song += " EOF"
        # remove '(*)'
        modified_song = re.sub(r"\([^()]*\)", "", song)
        modified_song = modified_song.replace("chorus", "").replace('&', '\n').lower()
        regex = re.compile('[%s]' % re.escape(string.punctuation.replace("'", "")))
        modified_song = regex.sub('', modified_song)
        df.loc[i, 'lyrics'] = modified_song
    return df


def create_x_y(songs, tokenize, total_words, max_len) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_sequences = []
    midi_meta = []
    for index, row in songs.iterrows():
        line = row['lyrics']
        token_list = tokenize.texts_to_sequences([line])[0]
        midi_metadata = row.drop(['artist', 'title', 'lyrics', 'filename']).values
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            midi_meta.append(midi_metadata)
    print(input_sequences[:10])

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding='pre'))
    x = input_sequences[:, :-1]
    x_midi = np.array(midi_meta)
    x_midi = np.where(x_midi == "unknown", -1, x_midi)
    y = input_sequences[:, -1]
    y = utils.to_categorical(y, num_classes=total_words)
    return x, x_midi, y


def split_df_to_train_val(df, ratio):
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_val = df[~mask]
    return df_train, df_val


def remove_words(words: List[str], songs):
    resulting_songs = []
    for song in songs:
        result = song
        for word in words:
            result = result.replace(word, '')
        resulting_songs.append(result)
    return resulting_songs
