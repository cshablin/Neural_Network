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

    # the values must be scaled to the range [-1,1] to match the output of the generator model -> tanh
    def scaler(self, df: pd.DataFrame):
        for column in df:
            max = df[column].max()
            min = df[column].min()
            col = 2*((df[column] - min)/(max - min)) - 1
            df[column] = col
        return df

    def test_load_lyrics(self):
        # plot the model
        diab_df = self.load_data()
        # class is not part of sample description
        del diab_df['class']
        diab_df = self.scaler(diab_df)
        print(diab_df.describe())

    def test_train_gan(self):
        # plot the model
        diab_df = self.load_data()
        # class is not part of sample description
        del diab_df['class']
        diab_df = self.scaler(diab_df)
        gan = GAN()
        (d_losses, d_accuracies, g_losses) = gan.train(df=diab_df, epochs=500, batch_size=16)
        #plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
