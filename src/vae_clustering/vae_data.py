import os
import cv2
import numpy as np
import tensorflow as tf


class VAEData():

    #COL_DARK_BLUE = 'blue' #dark blue
    COL_LIGHT_BLUE = 'cyan' #'light_blue'
    COL_GREEN = 'green'
    COL_PINK = 'pink'
    COL_RED = 'red'
    COL_VIOLET = 'violet'
    COL_WHITE = 'white'
    COL_YELLOW = 'yellow'

    SHAPE1 = "2x1"
    SHAPE2 = "2x2"
    SHAPE3 = "2x4"

    def __init__(self, path, img_size=(32,32)):
        self.path = path
        self.img_size = img_size

    #    self.colors = [VAEData.COL_LIGHT_BLUE, VAEData.COL_GREEN,VAEData.COL_RED,
    #                   VAEData.COL_YELLOW, VAEData.COL_WHITE]
        self.colors = [VAEData.SHAPE1, VAEData.SHAPE2, VAEData.SHAPE3]

        self._get_train_data(path) #os.path.join(path, "train"))
        #self._get_train_data(path)
    #    self._get_test_data(os.path.join(path, "test"))
       # self._get_real_data(os.path.join(path, "real"))

    def _get_train_data(self, train_data_path):
        self.train_imgs = {}
        for c in self.colors:
            col_path = os.path.join(train_data_path, c)
            self.train_imgs[c] = self._read_img_folder(col_path)
        #self.train_imgs = self._read_img_folder(train_data_path)

#        _train_imgs = list()
#        for dir in os.listdir(train_data_path):
#            block_dir = os.path.join(train_data_path, dir)
#            for c_dir in os.listdir(block_dir):
#                col_dir = os.path.join(block_dir, c_dir)
#                _train_imgs.append(self._read_img_folder(col_dir))
#        self.train_imgs = np.concatenate(_train_imgs, 0)


    def _get_test_data(self, test_data_path):
        self.test_imgs = {}
        for c in self.colors:
            col_path = os.path.join(test_data_path, c)
            self.test_imgs[c] = self._read_img_folder(col_path)

    def _get_real_data(self, real_data_path):
        self.real_imgs = {}
        for c in self.colors:
            col_path = os.path.join(real_data_path, 'color_' + c)
            if os.path.isdir(col_path):
                self.real_imgs[c] = self._read_img_folder(col_path)

    def test_data_by_color(self, color):
        return self.test_imgs[color]

    def _read_img_folder(self, folder_path):
        num_imgs = len(os.listdir(folder_path))
        imgs = np.zeros(shape=[num_imgs, self.img_size[0], self.img_size[1], 3], dtype=np.uint8)
        for i, img in enumerate(os.listdir(folder_path)):
      #      print(folder_path)
            imgs[i] = cv2.resize(cv2.imread(os.path.join(folder_path, img)), self.img_size)
        return imgs




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
