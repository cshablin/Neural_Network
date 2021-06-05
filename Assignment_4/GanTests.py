import unittest
import pandas as pd

from scipy.io import arff



class DiabetesTestCase(unittest.TestCase):

    def load_data(self) -> pd.DataFrame:
        diab_arf = arff.loadarff('diabetes.arff')
        diab_df = pd.DataFrame(diab_arf[0])
        return diab_df

    def test_load_lyrics(self):
        diab_df = self.load_data()
        print(diab_df.head())

