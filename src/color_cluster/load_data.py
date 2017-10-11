import os
import cv2
import numpy as np
import tensorflow as tf


class SiameseData():

    # #COL_DARK_BLUE = 'blue' #dark blue
    COL_LIGHT_BLUE = 'cyan' #'light_blue'
    COL_GREEN = 'green'
    COL_PINK = 'pink'
    COL_RED = 'red'
    COL_VIOLET = 'violet'
    COL_WHITE = 'white'
    COL_YELLOW = 'yellow'

    def __init__(self, path, img_size=(32, 32)):
        self.path = path
        self.img_size = img_size

        self.colors = [SiameseData.COL_LIGHT_BLUE, SiameseData.COL_GREEN, SiameseData.COL_RED,
                       SiameseData.COL_YELLOW, SiameseData.COL_WHITE]
        self._get_train_data(path)

    def _get_train_data(self, train_data_path):
        self.train_imgs = {}
        for c in self.colors:
            col_path = os.path.join(train_data_path, c)
            self.train_imgs[c] = self._read_img_folder(col_path)
        self.train_set = [self.train_imgs[c] for c in self.colors]


    def _read_img_folder(self, folder_path):
        num_imgs = len(os.listdir(folder_path))
        imgs = np.zeros(shape=[num_imgs, self.img_size[0], self.img_size[1], 3], dtype=np.uint8)
        for i, img in enumerate(os.listdir(folder_path)):
      #      print(folder_path)
            imgs[i] = cv2.resize(cv2.imread(os.path.join(folder_path, img)), self.img_size)
        return imgs

    def test_data_by_color(self, color):
        return self.test_imgs[color]

    def create_random_batch(self, data_set, batch_size):
        list_idxs_1 = np.random.randint(0, len(data_set), [batch_size])
        list_idxs_2 = np.random.randint(0, len(data_set) * 2 - 1, [batch_size])
        copy_idx = list_idxs_2 >= len(data_set)
        list_idxs_2[copy_idx] = list_idxs_1[copy_idx]

        images = np.zeros([batch_size, 2, self.img_size[0], self.img_size[1], 3])
        targets = np.zeros(batch_size)

        for i in range(batch_size):
            images[i, 0] = data_set[list_idxs_1[i]][np.random.randint(len(data_set[list_idxs_1[i]]))]
            images[i, 1] = data_set[list_idxs_2[i]][np.random.randint(len(data_set[list_idxs_2[i]]))]
            targets[i] = 1 if list_idxs_1[i] == list_idxs_2[i] else 0

        print("True Samples", np.count_nonzero(targets) / batch_size, " ", end='')
        return images, targets


    @property
    def tf_train_dataset(self):
        imgs_as_tensor = tf.constant(self.train_imgs, dtype=tf.uint8)
        dataset = tf.contrib.data.Dataset.from_tensor_slices(imgs_as_tensor)
        dataset.shuffle(1000)
        return dataset


    @property
    def tf_test_dataset(self):
        imgs_as_tensor = tf.constant(self.test_imgs[self.colors[0]], dtype=tf.uint8)

        for i in range(1, len(self.colors)):
            imgs_as_tensor = tf.concat([imgs_as_tensor, tf.constant(self.test_imgs[self.colors[i]], dtype=tf.uint8)], 1)
        return tf.contrib.data.Dataset.from_tensor_slices(imgs_as_tensor)
