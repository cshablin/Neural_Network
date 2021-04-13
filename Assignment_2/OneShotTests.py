import unittest
from typing import Dict, Tuple, List, Any

from skimage import io
import os
# import tensorflow as tf
# from skimage import transform
# from skimage import data
import matplotlib.pyplot as plt
import numpy as np
# from skimage.color import rgb2gray
import random

from Assignment_2.OneShotModel import OneShotModel

random.seed(51)
np.random.seed(51)


def load_data(data_dir: str) -> Tuple[Dict[int, np.ndarray], np.ndarray]:  # Tuple[np.ndarray, np.ndarray]
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
    return label_2_images, np.array(labels)
    # return label_2_images, np.array(images), np.array(labels)


def load_data_2(data_dir: str) -> Tuple[Dict[str, Dict[int, np.ndarray]], np.ndarray]:  # Tuple[np.ndarray, np.ndarray]
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
        file_2_image = {}
        file_counter = 1
        for f in file_names:
            image = io.imread(f)
            # file_2_image[os.path.basename(f)] = image
            file_2_image[file_counter] = image
            images.append(image)
            images_for_dict.append(image)
            labels.append(directory_counter)
            file_counter += 1
        # label_2_images[directory_counter] = np.array(images_for_dict)
        # label_2_images[d] = np.array(images_for_dict)
        label_2_images[d] = file_2_image
        images_for_dict = []
        directory_counter += 1
    return label_2_images, np.array(labels)


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


def chose_all_possible_groups(arr, group_size):

    # A temporary array to
    # store all combination
    # one by one
    data = [0]*group_size
    n = len(arr)
    # Print all combination
    # using temprary array 'data[]'
    output = []
    return combination_util(arr, data, 0, n - 1, 0, group_size, output)


def combination_util(arr, data, start, end, index, group_size, output) -> List[List[Any]]:

    # Current combination is ready
    # to be printed, print it
    temp = []
    if index == group_size:
        for j in range(group_size):
            # print(data[j], end = " ")
            temp.append(data[j])
        output.append(temp)
        return
    i = start
    while i <= end and end - i + 1 >= group_size - index:
        data[index] = arr[i]
        combination_util(arr, data, i + 1, end, index + 1, group_size, output)
        i += 1
    return output


# call this every epoch
def prepare_paired_x_and_y_data_set(label_2_images: Dict[int, np.ndarray], x_size: int, twin_percentage: int = 50) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    prepare balanced data set by 1 - random choosing twin images from labels with more than 1 image
                                 2 - random choosing pairs from all labels, note the chance of choosing twice the same
                                 label is scarce so

    :param twin_percentage: percentage of paired images from same label
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

    samples_1 = list()
    samples_2 = list()
    labels = list()
    twin_counter = 0
    # create same labeled pairs
    for multi_image_label in multi_image_labels:
        single_labeled_images = label_2_images[multi_image_label]
        all_possible_pairs_per_label = chose_all_possible_groups(single_labeled_images, group_size=2)
        pairs_per_label_counter = 0
        for possible_pair in all_possible_pairs_per_label:
            samples_1.append(possible_pair[0] / 255)  # Do not flatten for convolution
            samples_2.append(possible_pair[1] / 255)
            labels.append(1)
            if pairs_per_label_counter > 9:
                break  # Memory issue to load all possible pairs
            pairs_per_label_counter += 1
            twin_counter += 1

    # we want data set to be constructed with 'different' pairs from same image collection so the net could discriminate naturaly
    # we want having images that are part of a 'different' pairs and also part of 'same' pair, similar idea to "Triple Loss" in Face recognition
    random_image_labels_sample_1 = np.random.choice(multi_image_labels, int(twin_counter * ((100 - twin_percentage) / 100.0)))
    random_image_labels_sample_2 = np.random.choice(multi_image_labels, int(twin_counter * ((100 - twin_percentage) / 100.0)))
    # random_multi_image_labels = np.random.choice(multi_image_labels, int(x_size * (twin_percentage / 100.0)))



    # create same labeled pairs
    # for label in random_multi_image_labels:
    #     images = label_2_images[label]
    #
    #     indices = np.random.choice(images.shape[0], 2, replace=False)
    #
    #     # twin_image = np.random.choice(images[indices], 2)
    #     twin_image = images[indices]
    #     twin_image = twin_image.reshape((2, 250**2)) / 255
    #     samples_1.append(twin_image[0])
    #     samples_2.append(twin_image[1])
    #     labels.append(1)

    # create different labeled pairs
    for label_1, label_2 in zip(random_image_labels_sample_1, random_image_labels_sample_2):
        images_1 = label_2_images[label_1]
        images_2 = label_2_images[label_2]
        index_1 = np.random.choice(images_1.shape[0], 1, replace=False)
        index_2 = np.random.choice(images_2.shape[0], 1, replace=False)
        samples_1.append(images_1[index_1][0] / 255)
        samples_2.append(images_2[index_2][0] / 255)
        # samples_1.append(images_1[index_1][0].reshape((250**2)) / 255)
        # samples_2.append(images_2[index_2][0].reshape((250**2)) / 255)
        if label_1 != label_2:
            labels.append(0)
        else:
            labels.append(1)

    return [np.array(samples_1), np.array(samples_2)], np.array(labels)


def prepare_x_y_according_to_description(label_2_images: Dict[str, Dict[int, np.ndarray]], file: str) -> Tuple[List[np.ndarray], np.ndarray]:
    samples_1_test = list()
    samples_2_test = list()
    labels_test = list()
    with open(file, 'r+') as f:
        train_lines = f.readlines()
        num_of_samples = int(train_lines[0].strip())
        for index in range(1, num_of_samples + 1, 1):
            line = train_lines[index]
            words = line.split()
            image1 = label_2_images[words[0]][int(words[1])]
            image2 = label_2_images[words[0]][int(words[2])]
            samples_1_test.append(image1.reshape(250**2) / 255)  # Do not flatten for convolution
            samples_2_test.append(image2.reshape(250**2) / 255)
            labels_test.append(1)
        for index in range(num_of_samples + 1, 2 * num_of_samples + 1, 1):
            line = train_lines[index]
            words = line.split()
            image1 = label_2_images[words[0]][int(words[1])]
            image2 = label_2_images[words[2]][int(words[3])]
            samples_1_test.append(image1.reshape(250**2) / 255)  # Do not flatten for convolution
            samples_2_test.append(image2.reshape(250**2) / 255)
            labels_test.append(0)
    return [np.array(samples_1_test), np.array(samples_2_test)], np.array(labels_test)


class LoadDataTests(unittest.TestCase):

    data_folder = "Data"
    all_label_2_images, labels = load_data_2(data_folder)

    def test_all_data_set_analysis(self):
        num_of_classes = len(self.all_label_2_images)
        print("total classes size ", num_of_classes)
        print("total samples size ", self.labels.size)
        print("every sample is of shape ", self.all_label_2_images["Aaron_Eckhart"][1].shape)

        # plot number of examples per class
        plt.hist(self.labels, num_of_classes)
        plt.xlabel("class")
        plt.ylabel("num of images")
        plt.ylim([0, 65])
        plt.show()

    @unittest.skip("testing skipping")
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

    @unittest.skip("testing skipping")
    def test_prepare_data_set_for_model(self):
        batch_size = 32
        train, test = split_train_test(self.all_label_2_images, test_ratio=0.4)
        x, y = prepare_paired_x_and_y_data_set(test, x_size=batch_size)
        self.assertEqual(x[0].shape[0], 3402)
        self.assertEqual(x[0].shape[1], 250)
        self.assertEqual(x[0].shape[2], 250)

    def test_prepare_data_set_according_to_files(self):
        x_train, y_train = prepare_x_y_according_to_description(self.all_label_2_images, "pairsDevTrain.txt")
        x_test, y_test = prepare_x_y_according_to_description(self.all_label_2_images, "pairsDevTest.txt")
        self.assertEqual(x_train[0].shape[0], 2 * 1100)
        self.assertEqual(x_train[0].shape[1], 250)
        self.assertEqual(x_train[0].shape[2], 250)
        self.assertEqual(x_test[0].shape[0], 2 * 500)
        self.assertEqual(x_test[0].shape[1], 250)
        self.assertEqual(x_test[0].shape[2], 250)


class ModelTest(unittest.TestCase):


    def test_model(self):
        all_label_2_images, labels = load_data_2("Data")
        x_train, y_train = prepare_x_y_according_to_description(all_label_2_images, "pairsDevTrain.txt")
        # np.save( 'x1.npy', x_train[0] )
        # np.save( 'x2.npy', x_train[1] )
        # np.save( 'y.npy' , y_train )
        # X1 = np.load( 'x1.npy', allow_pickle=True)
        # X2 = np.load( 'x2.npy', allow_pickle=True)
        # Y = np.load( 'y.npy', allow_pickle=True)
        model = OneShotModel()
        parameters = {
            'batch_size' : 8 ,
            'validation_split' : 0.2 ,
            'epochs' : 9 ,
            'val_data' : None
        }
        h = model.fit(x_train, y_train, parameters)
        # h = model.fit([X1, X2], Y, parameters)
        model.plot_metric(h)

