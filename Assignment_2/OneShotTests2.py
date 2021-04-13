import unittest
from typing import Dict, Tuple, List, Any
from OneShotModel import OneShotModel

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


def load_set(file_name: str, data_dir: str):  # -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    with open(file_name) as pairsFile:
        lines = pairsFile.readlines()
        pairs_count = int(lines[0])
        # same pairs
        paired_images = []
        paired_labels = np.ones((pairs_count, 1), dtype=np.int8)  # 1 = same person
        for index in range(1, pairs_count + 1):
            dir_name, img1_name, img2_name = lines[index].strip().split("\t")
            dir_path = os.path.join(data_dir, dir_name)
            img1 = io.imread(os.path.join(dir_path, f"{dir_name}_{img1_name.zfill(4)}.jpg"))  # / 255
            img2 = io.imread(os.path.join(dir_path, f"{dir_name}_{img2_name.zfill(4)}.jpg"))  # / 255
            paired_images.append([img1, img2])

        # different pairs
        unpaired_images = []
        unpaired_labels = np.zeros((pairs_count, 1), dtype=np.int8)  # 0 = different person
        for index in range(1 + pairs_count, 2 * pairs_count + 1):
            dir1_name, img1_name, dir2_name, img2_name = lines[index].strip().split("\t")
            dir1_path = os.path.join(data_dir, dir1_name)
            dir2_path = os.path.join(data_dir, dir2_name)
            img1 = io.imread(os.path.join(dir1_path,
                                          f"{dir1_name}_{img1_name.zfill(4)}.jpg"))  # / 255 - divide by 255 only batch, memory saving
            img2 = io.imread(os.path.join(dir2_path, f"{dir2_name}_{img2_name.zfill(4)}.jpg"))  # / 255
            unpaired_images.append([img1, img2])

    return np.array(paired_images), paired_labels, np.array(unpaired_images), unpaired_labels


def split_set(set: np.ndarray, labels: np.ndarray, ratio: float = 0.2):
    size = set.shape[0]
    sub_set_indexes = np.random.choice(size, int(size * ratio), replace=False)
    sub_set = np.take(set, sub_set_indexes, axis=0).copy()
    seb_set_labels = np.take(labels, sub_set_indexes, axis=0).copy()
    return np.delete(set, sub_set_indexes, axis=0), np.delete(labels, sub_set_indexes, axis=0), sub_set, seb_set_labels


def shuffle_set(set: np.ndarray, labels: np.ndarray):  # reshuffle paired and unpaired sets between training epochs
    p = np.random.permutation(set.shape[0])
    return set[p], labels[p]


def get_batch(p_set: np.ndarray, up_set: np.ndarray, batch_size: int, offset: int = 0):
    batch_size = np.minimum(p_set.shape[0] - offset, batch_size)
    images = np.concatenate([p_set[offset:offset + batch_size], up_set[offset:offset + batch_size]])
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])  # 1 = same person

    return shuffle_set(images, labels)


class LoadDataTests(unittest.TestCase):
    data_folder = "Data"

    def test_load_set(self):
        file_name = "pairsDevTrain.txt"
        train_paired_images, train_paired_labels, train_unpaired_images, train_unpaired_labels = load_set(file_name,
                                                                                                          self.data_folder)
        self.assertEqual(train_paired_images.shape[0], 1100)
        self.assertEqual(train_paired_images.shape[1], 2)
        self.assertEqual(train_paired_images.shape[2], 250)
        self.assertEqual(train_paired_images.shape[3], 250)
        self.assertEqual(train_paired_images.shape[0], 1100)

        self.assertEqual(train_unpaired_images.shape[0], 1100)
        self.assertEqual(train_unpaired_images.shape[1], 2)
        self.assertEqual(train_unpaired_images.shape[2], 250)
        self.assertEqual(train_unpaired_images.shape[3], 250)
        self.assertEqual(train_unpaired_images.shape[0], 1100)

    def test_split_train_set(self):
        file_name = "pairsDevTrain.txt"
        ratio = 0.2
        train_paired_images, train_paired_labels, train_unpaired_images, train_unpaired_labels = load_set(file_name,
                                                                                                          self.data_folder)
        train_paired_images, train_paired_labels, validation_paired_images, validation_paired_labels = split_set(
            train_paired_images, train_paired_labels, ratio)
        train_unpaired_images, train_unpaired_labels, validation_unpaired_images, validation_unpaired_labels = split_set(
            train_unpaired_images, train_unpaired_labels, ratio)
        self.assertEqual(train_paired_images.shape[0], int(1100 * (1 - ratio)))
        self.assertEqual(train_paired_images.shape[1], 2)
        self.assertEqual(train_paired_images.shape[2], 250)
        self.assertEqual(train_paired_images.shape[3], 250)
        self.assertEqual(train_paired_images.shape[0], int(1100 * (1 - ratio)))
        self.assertEqual(validation_paired_images.shape[0], int(1100 * ratio))
        self.assertEqual(validation_paired_images.shape[1], 2)
        self.assertEqual(validation_paired_images.shape[2], 250)
        self.assertEqual(validation_paired_images.shape[3], 250)
        self.assertEqual(validation_paired_images.shape[0], int(1100 * ratio))

        self.assertEqual(train_unpaired_images.shape[0], int(1100 * (1 - ratio)))
        self.assertEqual(train_unpaired_images.shape[1], 2)
        self.assertEqual(train_unpaired_images.shape[2], 250)
        self.assertEqual(train_unpaired_images.shape[3], 250)
        self.assertEqual(train_unpaired_images.shape[0], int(1100 * (1 - ratio)))
        self.assertEqual(validation_unpaired_images.shape[0], int(1100 * ratio))
        self.assertEqual(validation_unpaired_images.shape[1], 2)
        self.assertEqual(validation_unpaired_images.shape[2], 250)
        self.assertEqual(validation_unpaired_images.shape[3], 250)
        self.assertEqual(validation_unpaired_images.shape[0], int(1100 * ratio))

    def test_shuffle(self):
        file_name = "pairsDevTest.txt"
        ratio = 0.2
        train_paired_images, train_paired_labels, train_unpaired_images, train_unpaired_labels = load_set(file_name,
                                                                                                          self.data_folder)
        train_paired_images, train_paired_labels, validation_paired_images, validation_paired_labels = split_set(
            train_paired_images, train_paired_labels, ratio)
        shape = validation_paired_images.shape
        e0 = np.sum(validation_paired_images[0])
        e10 = np.sum(validation_paired_images[10])
        validation_paired_images, validation_paired_labels = shuffle_set(validation_paired_images,
                                                                         validation_paired_labels)
        self.assertEqual(shape, validation_paired_images.shape)
        self.assertNotEqual(e0, np.sum(validation_paired_images[0]))
        self.assertNotEqual(e10, np.sum(validation_paired_images[10]))

    def test_get_batch(self):
        file_name = "pairsDevTest.txt"
        ratio = 0.2
        train_paired_images, train_paired_labels, train_unpaired_images, train_unpaired_labels = load_set(file_name,
                                                                                                          self.data_folder)
        train_paired_images, train_paired_labels, validation_paired_images, validation_paired_labels = split_set(
            train_paired_images, train_paired_labels, ratio)
        batch_images, batch_labels = get_batch(train_paired_images, train_unpaired_images, batch_size=20, offset=0)
        self.assertEqual(batch_images.shape[0], 40)
        self.assertEqual(batch_labels.shape[0], 40)
        self.assertEqual(np.sum(batch_labels), 20)

