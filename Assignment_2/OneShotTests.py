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
np.random.seed(51)


def load_data(data_dir: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:  # Tuple[np.ndarray, np.ndarray]
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
        label_2_images[directory_counter] = np.array(images_for_dict)
        images_for_dict = []
        directory_counter += 1
    return label_2_images, np.array(images), np.array(labels)
    # return np.array(images), np.array(labels)


def split_train_test(label_2_images: Dict[int, np.ndarray], test_ratio: float = 0.3) \
        -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
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


# call this every epoch
def prepare_paired_x_and_y_data_set(label_2_images: Dict[int, np.ndarray], x_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """

    :param x_size: number of samples in output data set
    :param label_2_images:  dictionary with keys of int labels representing folders of images and each value is a list of
     images corresponding to it's label/directory
    :return: data set [X1, X2], Y
    """
    multi_image_labels = []
    single_image_labels = []
    all_labels = []
    for label in label_2_images.keys():
        if len(label_2_images[label]) > 1:
            multi_image_labels.append(label)
        else:
            single_image_labels.append(label)
        all_labels.append(label)

    print("first image set shape", label_2_images[0].shape)
    all_labels = np.array(all_labels)
    multi_image_labels = np.array(multi_image_labels)

    random_image_labels_sample_1 = np.random.choice(all_labels, x_size // 2)
    random_image_labels_sample_2 = np.random.choice(all_labels, x_size // 2)
    random_multi_image_labels = np.random.choice(multi_image_labels, x_size // 2)

    samples_1 = list()
    samples_2 = list()
    labels = list()

    # create same labeled pairs
    for label in random_multi_image_labels:
        images = label_2_images[label]

        indices = np.random.choice(images.shape[0], 2, replace=False)

        # twin_image = np.random.choice(images[indices], 2)
        twin_image = images[indices]
        twin_image = twin_image.reshape((2, 250**2)) / 255
        samples_1.append(twin_image[0])
        samples_2.append(twin_image[1])
        labels.append(1)

    # create different labeled pairs
    for label_1, label_2 in zip(random_image_labels_sample_1, random_image_labels_sample_2):
        images_1 = label_2_images[label_1]
        images_2 = label_2_images[label_2]
        index_1 = np.random.choice(images_1.shape[0], 1, replace=False)
        index_2 = np.random.choice(images_2.shape[0], 1, replace=False)
        samples_1.append(images_1[index_1][0].reshape((250**2)) / 255)
        samples_2.append(images_2[index_2][0].reshape((250**2)) / 255)
        if label_1 != label_2:
            labels.append(0)
        else:
            labels.append(1)

    return [np.array(samples_1), np.array(samples_2)], np.array(labels)


class LoadDataTests(unittest.TestCase):

    data_folder = "Data"
    all_label_2_images, images, labels = load_data(data_folder)

    def test_all_data_set_analysis(self):
        num_of_classes = len(self.all_label_2_images)
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

        train, test = split_train_test(self.all_label_2_images, test_ratio=0.4)
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

    def test_prepare_data_set_for_model(self):
        batch_size = 32
        X, Y = prepare_paired_x_and_y_data_set(self.all_label_2_images, x_size=batch_size)
        self.assertEqual(np.sum(Y), batch_size / 2)
