import os

import  numpy as np
import tensorflow as tf

from color_cluster.config import *


class Model:

    def __init__(self, img_size, data=None, learning_rate=1e-3):
        self.data = data
        self.img_size = img_size
        self.learning_rate = learning_rate

        tf.reset_default_graph()
        self._build_inputs()
        self._build_graph()
        self._build_optimizer()
        self._init_session()

    def _build_inputs(self):
        self.img1_in = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 3])
        self.img2_in = tf.placeholder(tf.float32, shape=[None, self.img_size[0], self.img_size[1], 3])
        self.target_in = tf.placeholder(tf.float32, shape=[None, 1])

    def _build_graph(self):
        def convolutions(img, reuse):
            with tf.variable_scope("c1"):
                h1_conv = tf.layers.conv2d(img,
                                           filters=CONV1_NUM_FILTERS,
                                           kernel_size=CONV1_FILTER_SIZE,
                                           padding=PADDING,
                                           strides=CONV1_STRIDE,
                                           activation=ACTIVATION_FN,
                                           kernel_initializer=KERNEL_INITIALIZER,
                                           bias_initializer=BIAS_INITIALIZER,
                                           reuse=reuse)
                h1 = POOL_FN(h1_conv, pool_size=POOL1_FILTER_SIZE, strides=POOL1_STRIDE, padding=PADDING)

            with tf.variable_scope("c2"):
                h2_conv = tf.layers.conv2d(h1,
                                           filters=CONV2_NUM_FILTERS,
                                           kernel_size=CONV2_FILTER_SIZE,
                                           padding=PADDING,
                                           strides=CONV2_STRIDE,
                                           activation=ACTIVATION_FN,
                                           kernel_initializer=KERNEL_INITIALIZER,
                                           bias_initializer=BIAS_INITIALIZER,
                                           reuse=reuse)
                h2 = POOL_FN(h2_conv, pool_size=POOL2_FILTER_SIZE, strides=POOL2_STRIDE, padding=PADDING)

            with tf.variable_scope("c3"):
                h3_conv = tf.layers.conv2d(h2,
                                           filters=CONV3_NUM_FILTERS,
                                           kernel_size=CONV3_FILTER_SIZE,
                                           padding=PADDING,
                                           strides=CONV3_STRIDE,
                                           activation=ACTIVATION_FN,
                                           kernel_initializer=KERNEL_INITIALIZER,
                                           bias_initializer=BIAS_INITIALIZER,
                                           reuse=reuse)
                h3 = POOL_FN(h3_conv, pool_size=POOL3_FILTER_SIZE, strides=POOL3_STRIDE, padding=PADDING)
            h3_flat = tf.contrib.layers.flatten(h3)

            with tf.variable_scope("d4"):
                h4 = tf.layers.dense(h3_flat,
                                     units=DENSE4_NUM_OUTPUTS,
                                     activation=ACTIVATION_FN,
                                     kernel_initializer=WEIGHT_INITIALIZER,
                                     bias_initializer=BIAS_INITIALIZER,
                                     reuse=reuse)

            with tf.variable_scope("d5"):
                h5 = tf.layers.dense(h4,
                                     units=LATENT_DIMENSION,
                                     kernel_initializer=WEIGHT_INITIALIZER,
                                     bias_initializer=BIAS_INITIALIZER,
                                     reuse=reuse)
            return h5

        with tf.variable_scope("convolutions") as conv_scope:
            self.h_img1 = convolutions(self.img1_in / 255, False)
            conv_scope.reuse_variables()
            h_img2 = convolutions(self.img2_in / 255, True)

        combined = tf.abs(self.h_img1 - h_img2)
        tanh_c = tf.tanh(combined)
        self.prediction = tf.reduce_mean(1 - tf.maximum(tanh_c, -tanh_c), -1, keep_dims=True)

    @staticmethod
    def _binary_cross_entropy(target, prediction, epsilon=1e-8):
        """Binary Cross Entropy between true and predicted image"""
        prediction = tf.maximum(epsilon, tf.minimum(prediction, 1 - epsilon))
        point_wise_error = - (target * tf.log(prediction + epsilon) + (1-target) * tf.log(1 - prediction + epsilon))
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=[ -1])
        return tf.reduce_mean(sample_wise_error)

    def _build_optimizer(self):
        self.loss = self._binary_cross_entropy(self.target_in, self.prediction)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, batch_size, num_batches, verbose_interval=10):
        if self.data is None:
            raise ValueError("No data object feed -> training not possible")
        else:
            print("Start training")
        for i in range(num_batches):

            imgs, targets = self.data.create_random_batch(self.data.train_set, batch_size)
            feed_dict = {self.img1_in: imgs[:, 0],
                         self.img2_in: imgs[:, 1],
                         self.target_in: np.expand_dims(targets, -1)}

            _, loss = self.sess.run(fetches=(self.optimizer, self.loss), feed_dict=feed_dict)
            print("Saw", i, "batches. Loss:", loss)
            if i % verbose_interval == 0:
                self.eval(batch_size=batch_size, num_batches=10)

    def get_latent(self, imgs):
        return self.sess.run(self.h_img1, {self.img1_in: imgs})

    def eval(self, batch_size, num_batches):
        avg_loss = 0
        avg_accuracy = 0
        avg_false_pos = 0
        avg_false_neg = 0
        for _ in range(num_batches):
            imgs, targets = self.data.create_random_batch(self.data.train_set, batch_size)
            feed_dict = {self.img1_in: imgs[:, 0],
                         self.img2_in: imgs[:, 1],
                         self.target_in: np.expand_dims(targets, -1)}
            loss, predictions = self.sess.run(fetches=(self.loss, self.prediction), feed_dict=feed_dict)

            predictions = np.squeeze(predictions)
            avg_loss += loss / num_batches

            predictions[predictions > 0.5] = 1
            predictions[predictions <= 0.5] = 0
            print(np.all(predictions == 0))
            positives = np.count_nonzero(targets==1)
            negatives = np.count_nonzero(targets==0)
            print(positives, negatives)
            cur_acc = np.count_nonzero(predictions == targets) / batch_size
            fp = np.count_nonzero(predictions > targets)
            fn = np.count_nonzero(targets > predictions)
            print(fp, fn)
            cur_false_pos = np.count_nonzero(predictions > targets) / positives
            cur_false_neg = np.count_nonzero(targets > predictions) / negatives #  np.count_nonzero(targets)
            avg_accuracy += cur_acc / num_batches
            avg_false_pos += cur_false_pos / num_batches
            avg_false_neg += cur_false_neg / num_batches

        print("Evaluation:, Loss:", avg_loss,
              "Accuracy:", avg_accuracy,
              "False Negative", avg_false_neg,
              "False Positive", avg_false_pos)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path)

    def load_model(self, path):
        """
        load the model

       :param path: the path to the save location
        """
        saver = tf.train.import_meta_graph(path + '/.meta')
        saver.restore(self.sess, path)
