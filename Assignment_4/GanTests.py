import unittest
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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

    def show_plot(self, arr: np.ndarray, title):
        x = range(arr.shape[0])
        plt.title(title)
        plt.xlabel("Epochs")
        plt.plot(x, arr)
        plt.show()

    def test_load(self):
        # plot the model
        diab_df = self.load_data()
        # class is not part of sample description
        del diab_df['class']
        diab_df = self.scaler(diab_df)
        print(diab_df.describe())

    def test_summery_gan(self):
        gan = GAN(10, 8, [30, 15], [32, 16], 'tanh')

    def test_train_gan(self):
        # plot the model
        diab_df = self.load_data()
        # class is not part of sample description
        del diab_df['class']
        diab_df = self.scaler(diab_df)
        gan = GAN(10, 8 , [30, 15], [32, 16], 'tanh')
        d_losses, d_accuracies, g_losses, g_accuracies, d_fake_losses, d_real_losses, d_fake_accuracies, d_real_accuracies = gan.train(df=diab_df, epochs=500, batch_size=16)
        self.show_plot(d_losses, 'd_losses')
        self.show_plot(d_accuracies,'d_accuracies')
        self.show_plot(g_losses, 'g_losses')
        self.show_plot(g_accuracies, 'g_accuracies')

        self.plot_metric(d_losses, d_fake_accuracies, d_real_accuracies)

        # Test Discriminator on real  data
        X_train = diab_df.values
        y_real = np.ones((X_train.shape[0], 1))
        _, acc_real = gan.discriminator.evaluate(X_train, y_real, verbose=1)
        print('Evaluate real ', acc_real)

        #plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

    def plot_metric(self, d_losses: np.ndarray, d_fake_accuracies: np.ndarray, d_real_accuracies: np.ndarray) -> None:
        import matplotlib.pyplot as plt

        epochs = range(1, d_losses.shape[0] + 1)
        plt.plot(epochs, d_losses)
        plt.plot(epochs, d_fake_accuracies)
        plt.plot(epochs, d_real_accuracies)
        plt.title('Gan')
        plt.xlabel("Epochs")
        plt.legend(['d_loss', 'd_fake_accuracies', 'd_real_accuracies'])
        plt.show()

    def plot_metric_general(self, graphs: List[np.ndarray], labels: List[str]) -> None:
        import matplotlib.pyplot as plt
        epochs = range(1, graphs[0].shape[0] + 1)
        plt.xlabel("Epochs")
        plt.title('Gan')
        for i in range(len(labels)):
            plt.plot(epochs, graphs[i])
        plt.legend(labels)
        plt.show()


class CreditTestCase(unittest.TestCase):

    def load_data(self) -> pd.DataFrame:
        diab_arf = arff.loadarff('german_credit.arff')
        diab_df = pd.DataFrame(diab_arf[0])
        return diab_df

    def test_load(self):
        df = self.prepare()
        df = df.astype('float')

        print(df.head())

    def prepare(self):
        cred_df = self.load_data()
        y = cred_df.pop('21')
        # print(cred_df.describe())
        cat_labels = ['1', '3', '4', '6', '7', '9', '10', '12', '14', '15', '17', '19', '20']
        # One hot encoding
        df = pd.get_dummies(cred_df, columns=cat_labels)
        df = df.astype('float')

        # numeric scaling
        scale = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scale.fit_transform(df)
        df.loc[:, :] = scaled_values

        # scale = MinMaxScaler(feature_range=(-1, 1))
        # df['2'] = scale.fit_transform(df['2'].values.reshape(-1, 1))
        # df['5'] = scale.fit_transform(df['5'].values.reshape(-1, 1))
        # df['8'] = scale.fit_transform(df['8'].values.reshape(-1, 1))
        # df['11'] = scale.fit_transform(df['11'].values.reshape(-1, 1))
        # df['13'] = scale.fit_transform(df['13'].values.reshape(-1, 1))
        # df['16'] = scale.fit_transform(df['16'].values.reshape(-1, 1))
        # df['18'] = scale.fit_transform(df['18'].values.reshape(-1, 1))
        return df

    def test_credit(self):
        df_credit = self.prepare()
        # df_credit = df_credit.astype('float')

        # gan = GAN(20, 61, [60, 30], [64, 32], 'sigmoid')
        gan = GAN(30, 61, [128, 256, 512], [512, 256, 128], 'tanh')
        d_losses, d_accuracies, g_losses, g_accuracies, d_fake_losses, d_real_losses, d_fake_accuracies, d_real_accuracies = gan.train(df=df_credit, epochs=500, batch_size=16)
        self.show_plot(d_losses, 'd_losses')
        self.show_plot(d_accuracies,'d_accuracies')
        self.show_plot(g_losses, 'g_losses')
        self.show_plot(g_accuracies, 'g_accuracies')

        self.plot_metric(d_losses, d_fake_accuracies, d_real_accuracies)

    def plot_metric(self, d_losses: np.ndarray, d_fake_accuracies: np.ndarray, d_real_accuracies: np.ndarray) -> None:
        import matplotlib.pyplot as plt

        epochs = range(1, d_losses.shape[0] + 1)
        plt.plot(epochs, d_losses)
        plt.plot(epochs, d_fake_accuracies)
        plt.plot(epochs, d_real_accuracies)
        plt.title('Gan')
        plt.xlabel("Epochs")
        plt.legend(['d_loss', 'd_fake_accuracies', 'd_real_accuracies'])
        plt.show()

    def plot_metric_general(self, graphs: List[np.ndarray], labels: List[str]) -> None:
        import matplotlib.pyplot as plt
        epochs = range(1, graphs[0].shape[0] + 1)
        plt.xlabel("Epochs")
        plt.title('Gan')
        for i in range(len(labels)):
            plt.plot(epochs, graphs[i])
        plt.legend(labels)
        plt.show()

    def show_plot(self, arr: np.ndarray, title):
        x = range(arr.shape[0])
        plt.title(title)
        plt.xlabel("Epochs")
        plt.plot(x, arr)
        plt.show()