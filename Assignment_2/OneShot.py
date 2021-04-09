import unittest
from typing import Dict, Tuple, List

from skimage import io
import os
# import tensorflow as tf
# from skimage import transform
# from skimage import data
import matplotlib.pyplot as plt
import numpy as np
# from skimage.color import rgb2gray
import random
random.seed(51)


def load_data(data_dir: str) -> Tuple[Dict[int, List[np.ndarray]], np.ndarray, np.ndarray]:  # Tuple[np.ndarray, np.ndarray]
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    images_for_dict = []
    label_2_images = {}
    directory_counter = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            image = io.imread(f)
            images.append(image)
            images_for_dict.append(image)
            labels.append(directory_counter)
        label_2_images[directory_counter] = images_for_dict
        images_for_dict = []
        directory_counter += 1
    return label_2_images, np.array(images), np.array(labels)
    # return np.array(images), np.array(labels)


def split_train_test(label_2_images: Dict[int, List[np.ndarray]], test_ratio: float = 0.3) \
        -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    """

    :param label_2_images: dictionary with keys of int labels representing folders of images and each value is a list of
     images corresponding to it's label/directory
    :param test_ratio: represent the percentage of all labels for test phase
    :return: train abel_2_images dictionary and test abel_2_images dictionary
    """

    key_list = list(label_2_images.keys())
    test_key_count = int(len(key_list) * test_ratio)
    test_keys = [random.choice(key_list) for ele in range(test_key_count)]
    train_keys = [ele for ele in key_list if ele not in test_keys]

    testing_dict = dict((key, label_2_images[key]) for key in test_keys
                        if key in label_2_images)
    training_dict = dict((key, label_2_images[key]) for key in train_keys
                         if key in label_2_images)
    return training_dict, testing_dict


class LoadDataTests(unittest.TestCase):

    data_folder = "Data"
    label_2_images, images, labels = load_data(data_folder)

    def test_all_data_set_analysis(self):
        num_of_classes = len(self.label_2_images)
        print("total classes size ", num_of_classes)
        print("total samples size ", self.labels.size)
        print("every sample is of shape ", self.images[0].shape)

        # plot number of examples per class
        plt.hist(self.labels, num_of_classes)
        plt.xlabel("class")
        plt.ylabel("num of images")
        plt.ylim([0, 65])
        plt.show()

    def test_splitting_data_sets_analysis(self):

        train, test = split_train_test(self.label_2_images, test_ratio=0.4)
        validation, test1 = split_train_test(test, test_ratio=0.5)

        print("# train ", len(train))
        print("# validation ", len(validation))
        print("# test ", len(test))

        train_labels = []
        for train_l, images in train.items():
            for _ in images:
                train_labels.append(train_l)
        plt.hist(train_labels, len(train))
        plt.xlabel("class in train set")
        plt.ylabel("num of images")
        plt.ylim([0, 65])
        plt.show()

        val_labels = []
        for train_l, images in validation.items():
            for _ in images:
                val_labels.append(train_l)
        plt.hist(val_labels, len(train))
        plt.xlabel("class in validation set")
        plt.ylabel("num of images")
        plt.ylim([0, 65])
        plt.show()

        test_labels = []
        for train_l, images in test.items():
            for _ in images:
                test_labels.append(train_l)
        plt.hist(test_labels, len(train))
        plt.xlabel("class in test set")
        plt.ylabel("num of images")
        plt.ylim([0, 65])
        plt.show()


