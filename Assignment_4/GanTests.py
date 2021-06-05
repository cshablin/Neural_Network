import unittest
import pandas as pd

from scipy.io import arff
from tensorflow.python.keras.utils.vis_utils import plot_model

from Assignment_4.Gan import GAN


class DiabetesTestCase(unittest.TestCase):

    def load_data(self) -> pd.DataFrame:
        diab_arf = arff.loadarff('diabetes.arff')
        diab_df = pd.DataFrame(diab_arf[0])
        return diab_df

    def test_load_lyrics(self):
        # plot the model
        model = GAN()
        #plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

