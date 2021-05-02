import unittest

from Assignment_3.DataLoad import load_lyrics
from Assignment_3.MusicalRNN import LyricsGenerator


class MusicTestCase(unittest.TestCase):

    def test_load_lyrics(self):
        result = load_lyrics("Lyrics\\lyrics_train_set.csv")
        self.assertIsNotNone(result)

    def test_word_embedding(self):
        lyrics_generator = LyricsGenerator()
        lyrics_generator.embedding_spacy("")


if __name__ == '__main__':
    unittest.main()
