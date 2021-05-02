import unittest

from Assignment_3.DataLoad import load_lyrics


class MusicTestCase(unittest.TestCase):

    def test_load_lyrics(self):
        result = load_lyrics("Lyrics\\lyrics_train_set.csv")
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
