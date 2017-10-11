import os, datetime
import numpy as np
from vae_clustering.vae_config import *

class VAE():
    """
    Implementation of a Variational Autoencoder (VAE) (Auto-Encoding Variational Bayes, Kingma & Welling, 2013)
    """

    def __init__(self, input_size, latent_space_dim, train_loss):
        """
        Creates new variational autoencoder
        :param input_size: Dimensions of input images
        :param latent_space_dim: Dimension of latent space
        """
        self.input_size = list(input_size) + [3]
        self.latent_space_dim = latent_space_dim
        self.train_loss = train_loss

        print("Training on:", train_loss)

        prior_mean = tf.zeros(latent_space_dim)
        prior_std = 4 * tf.ones(latent_space_dim)
        self.prior = tf.contrib.distributions.MultivariateNormalDiag(prior_mean, prior_std)

        self._define_inputs()
        self._build_graph()
        self._initialize_session()


    def _define_inputs(self):
        with tf.name_scope('InOut'):
            self.images = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_size, name='image_ph')
            self.training = tf.placeholder_with_default(tf.constant(False), shape=[])


    def _build_graph(self):
        with tf.name_scope('Encoder'):
           # im_noise = tf.minimum(tf.maximum(0.0, self.images + tf.random_normal(tf.shape(self.images), stddev=15)),
           #                       255.0)
            ''' LAYER 1'''
            h1_conv = tf.contrib.layers.conv2d(self.images / 255,
                                               num_outputs=ENC_CONV1_NUM_FILTERS,
                                               kernel_size=ENC_CONV1_FILTER_SIZE,
                                               padding=ENC_PADDING,
            #                                   stride=ENC_POOL1_STRIDE,
                                               activation_fn=ACTIVATION_FN,
                                               normalizer_fn=NORMALIZATION_FN_CONV,
                                               weights_initializer=KERNEL_INITIALIZER,
                                               biases_initializer=BIAS_INITIALIZER)
            h1 = POOL_FN(h1_conv, kernel_size = ENC_POOL1_FILTER_SIZE, stride=ENC_POOL1_STRIDE, padding=ENC_PADDING)

            ''' LAYER 2'''
            h2_conv = tf.contrib.layers.conv2d(h1, #self.images / 255,
                                               num_outputs=ENC_CONV2_NUM_FILTERS,
                                               kernel_size=ENC_CONV2_FILTER_SIZE,
                                               padding=ENC_PADDING,
                                           #    stride=ENC_POOL2_STRIDE,
                                               activation_fn=ACTIVATION_FN,
                                               normalizer_fn=NORMALIZATION_FN_CONV,
                                               weights_initializer=KERNEL_INITIALIZER,
                                               biases_initializer=BIAS_INITIALIZER)
            h2 = POOL_FN(h2_conv, kernel_size = ENC_POOL2_FILTER_SIZE, stride=ENC_POOL2_STRIDE, padding=ENC_PADDING)

            ''' LAYER 3'''
            h3_conv = tf.contrib.layers.conv2d(h2,
                                               num_outputs=ENC_CONV3_NUM_FILTERS,
                                               kernel_size=ENC_CONV3_FILTER_SIZE,
                                               padding=ENC_PADDING,
                                            #   stride=ENC_POOL3_STRIDE,
                                               activation_fn=ACTIVATION_FN,
                                               normalizer_fn=NORMALIZATION_FN_CONV,
                                               weights_initializer=KERNEL_INITIALIZER,
                                               biases_initializer=BIAS_INITIALIZER)
            h3 = POOL_FN(h3_conv, kernel_size = ENC_POOL3_FILTER_SIZE, stride=ENC_POOL3_STRIDE, padding=ENC_PADDING)
            h3_flat = tf.contrib.layers.flatten(h3)
            print(h3.shape)

            ''' LAYER 4'''
            h4 = tf.contrib.layers.fully_connected(h3_flat,
                                                   num_outputs=ENC_DENSE4_NUM_OUTPUTS,
                                                   activation_fn=ACTIVATION_FN,
                                                   normalizer_fn=NORMALIZATION_FN_DENSE,
                                                   weights_initializer=WEIGHT_INITIALIZER,
                                                   biases_initializer=BIAS_INITIALIZER)
   #        im_noise = tf.minimum(tf.maximum(0.0, self.images + tf.random_normal(tf.shape(self.images), stddev=15)), 255.0)
#            in_flat = tf.contrib.layers.flatten(self.images / 255)
#            h1 = tf.contrib.layers.fully_connected(in_flat,
#                                                   num_outputs = 512,
#                                                   activation_fn=tf.nn.relu,
#                                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                                   biases_initializer=tf.zeros_initializer())
#            h2 = tf.contrib.layers.fully_connected(h1,
#                                                   num_outputs = 256,
#                                                   activation_fn=tf.nn.relu,
#                                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                                   biases_initializer=tf.zeros_initializer())

            ''' LAYER 5'''
        with tf.name_scope('Latent_Distribution'):
            print(self.latent_space_dim)
            self.latent_mean = tf.contrib.layers.fully_connected(h4,
                                                                 num_outputs=self.latent_space_dim,
                                                                 activation_fn=None,
                                                                 weights_initializer=WEIGHT_INITIALIZER,
                                                                 biases_initializer=BIAS_INITIALIZER)
        #    self.latent_mean = self.latent_mean / tf.norm(self.latent_mean, axis=-1, keep_dims=True)
            self.latent_std = tf.contrib.layers.fully_connected(h4,
                                                                num_outputs=self.latent_space_dim,
                                                                activation_fn=tf.exp,
                                                                weights_initializer=WEIGHT_INITIALIZER,
                                                                biases_initializer=BIAS_INITIALIZER)

            self.latent_sample = self._sample(self.latent_mean, self.latent_std)

            decoder_in = tf.cond(self.training, lambda: self.latent_sample, lambda: self.latent_mean)

        with tf.name_scope('Decoder'):
            ''' LAYER 6'''
#            h3 = tf.contrib.layers.fully_connected(decoder_in,
#                                                   num_outputs=256,
#                                                   activation_fn=tf.nn.relu,
#                                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                                   biases_initializer=tf.zeros_initializer())
#            h4 = tf.contrib.layers.fully_connected(h3,
#                                                   num_outputs=512,
#                                                  activation_fn=tf.nn.relu,
#                                                   weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                                   biases_initializer=tf.zeros_initializer())
#            out_flat = tf.contrib.layers.fully_connected(h4,
#                                                        num_outputs=3072,
#                                                        activation_fn=tf.nn.sigmoid,
#                                                        weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
#                                                        biases_initializer=tf.zeros_initializer())
#            self.reconstructed_img = tf.reshape(out_flat, [-1, 32, 32, 3])

            h6_flat = tf.contrib.layers.fully_connected(decoder_in,
                                                        num_outputs=(INITIAL_IMAGE_SIZE ** 2) * INITIAL_NUM_CHANNELS,
                                                        activation_fn=ACTIVATION_FN,
                                                        normalizer_fn=NORMALIZATION_FN_DENSE,
                                                        weights_initializer=WEIGHT_INITIALIZER,
                                                        biases_initializer=BIAS_INITIALIZER)

            h6 = tf.reshape(h6_flat, shape=[-1, INITIAL_IMAGE_SIZE, INITIAL_IMAGE_SIZE, INITIAL_NUM_CHANNELS])
            '''LAYER 7'''
            h7 = tf.contrib.layers.conv2d_transpose(h6,
                                                    num_outputs=DEC_CONV2_NUM_FILTERS,
                                                    kernel_size=DEC_CONV2_FILTER_SIZE,
                                                    padding=DEC_PADDING,
                                                    stride=DEC_CONV2_STRIDE,
                                                    activation_fn=ACTIVATION_FN,
                                                    normalizer_fn=NORMALIZATION_FN_CONV,
                                                    weights_initializer=WEIGHT_INITIALIZER,
                                                    biases_initializer=BIAS_INITIALIZER)
            '''LAYER 8'''
            h8 = tf.contrib.layers.conv2d_transpose(h7,
                                                    num_outputs=DEC_CONV3_NUM_FILTERS,
                                                    kernel_size=DEC_CONV3_FILTER_SIZE,
                                                    padding=DEC_PADDING,
                                                    stride=DEC_CONV3_STRIDE,
                                                    activation_fn=ACTIVATION_FN,
                                                    normalizer_fn=NORMALIZATION_FN_CONV,
                                                    weights_initializer=WEIGHT_INITIALIZER,
                                                    biases_initializer=BIAS_INITIALIZER)
            '''LAYER 9'''
            h9 = tf.contrib.layers.conv2d_transpose(h8,
                                                    num_outputs=DEC_CONV4_NUM_FILTERS,
                                                    kernel_size=DEC_CONV4_FILTER_SIZE,
                                                    padding=DEC_PADDING,
                                                    stride=DEC_CONV4_STRIDE,
                                                    activation_fn=ACTIVATION_FN,
                                                    normalizer_fn=NORMALIZATION_FN_CONV,
                                                    weights_initializer=WEIGHT_INITIALIZER,
                                                    biases_initializer=BIAS_INITIALIZER)
            '''Output Layer'''
            self.reconstructed_img = tf.contrib.layers.conv2d_transpose(h9,
                                                                        num_outputs=3,
                                                                        kernel_size=DEC_OUTPUT_FILTER_SIZE,
                                                                        padding=DEC_PADDING,
                                                                        stride=1,
                                                                        activation_fn=tf.nn.sigmoid,
                                                                        weights_initializer=WEIGHT_INITIALIZER,
                                                                        biases_initializer=BIAS_INITIALIZER)

            recon_std_log = tf.get_variable('recon', dtype=tf.float32, initializer=tf.constant(np.log(0.05).astype(np.float32)))
            recon_std = tf.exp(recon_std_log)
            #recon_std = tf.Print(recon_std, [recon_std])
            '''Loss'''
            self.reconstruction_loss_bce = self._binary_cross_entropy(self.images / 255, self.reconstructed_img)
            self.reconstruction_loss_nll = self._nll(self.images / 255, self.reconstructed_img, recon_std)
            self.regularization_loss = self._kl_loss(self.latent_mean, self.latent_std)

            if self.train_loss == 'bce':
                self.train_loss = self.reconstruction_loss_bce + self.regularization_loss
            elif self.train_loss == 'nll':
                self.train_loss = self.reconstruction_loss_nll + self.regularization_loss
            else:
                raise ValueError("Loss needs to be either bce or nll")

            '''Optimizer'''
            self.optimizer = tf.train.AdamOptimizer().minimize(self.train_loss)

    @staticmethod
    def _sample(mean, std):
        """Sample by reparameterization [Kingma & Welling 2013]"""
        eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0)
        return mean + std * eps

    def _kl_loss(self, mean, stddev):
        """KL(p||q) with given distribution p and prior q"""
        latent_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)
        return tf.reduce_mean(tf.contrib.distributions.kl_divergence(latent_dist, self.prior, allow_nan_stats=False))

    @staticmethod
    def _binary_cross_entropy(target, prediction, epsilon=1e-8):
        """Binary Cross Entropy between true and predicted image"""
        point_wise_error = - (target * tf.log(prediction + epsilon) + (1-target) * tf.log(1 - prediction + epsilon))
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=[-3, -2, -1])
        return tf.reduce_mean(sample_wise_error)

    def _nll(self, target, prediction_mean, prediction_std):
        const_term = 0.5 * tf.log(2 * np.pi)
        sig_term = 0.5 * tf.log(prediction_std**2)
        data_term = 0.5 *((target - prediction_mean) ** 2) / (prediction_std ** 2)
        point_wise_error = const_term + sig_term + data_term
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=[-3, -2, -1])
        return tf.reduce_mean(sample_wise_error)

    def _initialize_session(self):
        self.tf_session = tf.Session()
        self.tf_session.run(tf.global_variables_initializer())

        self.tf_saver = tf.train.Saver()

    def train(self, train_data, num_epochs, batch_size):
        """
        Trains the model
        :param train_data: tf.contrib.data.Dataset Object containing the train data, needs to take care of shuffle
        :param test_data: tf.contrib.data.Dataset Object containing the test data
        :param num_epochs: number of epochs to train
        :param batch_size: number of images per batch
        :return:
        """

        best_result = np.infty

        for epoch in range(num_epochs):
            # on the fly data generation

       #     print('Train epoch number {}'.format(epoch))
            train_reconstruction_loss_bce, train_reconstruction_loss_nll, train_regularization_loss =\
                self._epoch(train_data, batch_size, training=True)
            print('Train epoch number {} reconstruction_loss_bce = {}, reconstruction_loss_nll = {}, regularization_loss = {}'
                  .format(epoch, train_reconstruction_loss_bce, train_reconstruction_loss_nll , train_regularization_loss))

      #      test_reconstruction_loss, test_regularization_loss = self._epoch(test_data,  batch_size, training=False)
      #      print('Test: reconstruction_loss = {}, regularization_loss = {}'
      #            .format(test_reconstruction_loss, test_regularization_loss))
      #      total_test_loss = test_reconstruction_loss + test_regularization_loss

     #       if total_test_loss < best_result:
     #           print('Validation score has improved from {} to {} saving the model\n'
      #                .format(best_result, total_test_loss))
      #          best_result = total_test_loss
     #       else:
    #            print('Validation score has not improved\n')


    def _epoch(self, data, batch_size, training):
        reconstruction_loss_bce_accu = 0
        reconstruction_loss_nll_accu = 0
        regularization_loss_accu = 0
        nr_data = len(data)
        num_batches = int(np.floor(nr_data / batch_size))
   #     print(nr_data, num_batches)

        random_idx = np.random.permutation(nr_data)


#        nr_batches = 0


#        batches = data.batch(batch_size=batch_size)
#        iterator = tf.contrib.data.Iterator.from_structure(batches.output_tpyes, batches.output_shapes)
#        current_batch = iterator.get_next()
#        self.tf_session.run(iterator.make_initializer(batches))
        for i in range(num_batches):
        #while True:
         #   try:
         #   images = self.tf_session.run(current_batch)
            images = data[random_idx[i*batch_size: (i+1)*batch_size]]
            feed_dict = {self.images: images,
                         self.training: training}
            if training:
                _, reconstruction_loss_bce, reconstruction_loss_nll, regularization_loss = self.tf_session.run(
                    fetches=(self.optimizer, self.reconstruction_loss_bce, self.reconstruction_loss_nll, self.regularization_loss),
                    feed_dict=feed_dict)
            else:
                reconstruction_loss_bce, reconstruction_loss_nll, regularization_loss = self.tf_session.run(
                    fetches=(self.reconstruction_loss_bce, self.reconstruction_loss_nll, self.regularization_loss),
                    feed_dict=feed_dict)

#                nr_batches += 1
            reconstruction_loss_bce_accu += reconstruction_loss_bce
            reconstruction_loss_nll_accu += reconstruction_loss_nll
            regularization_loss_accu += regularization_loss
            #except tf.errors.OUT_OF_RANGE:
            #    break

        return reconstruction_loss_bce_accu / num_batches,\
               reconstruction_loss_nll_accu / num_batches, \
               regularization_loss_accu / num_batches



    def load(self, path):
        """
        load the model
        :param path: the path to the save location
        """
        self.tf_saver.restore(self.tf_session, path)
        print("Successfully load model from save path: %s" % path)

    def save(self, path):
        print("Saving model in %s".format(path))
        i = datetime.datetime.now()
        path = path+str(i.isoformat())+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        self.tf_saver.save(self.tf_session, path+"model")


    def get_latent_distribution(self, images, batch_size=-1):
        if batch_size == -1:
            means, _ = self.tf_session.run(fetches=(self.latent_mean, self.latent_std),
                                           feed_dict={self.images: images})
            return means
        else:
            num_batches = int(len(images) / batch_size)
            mean_list = list()
            for i in range(num_batches):
                means, _ = self.tf_session.run(fetches=(self.latent_mean, self.latent_std),
                                               feed_dict={self.images: images[i * batch_size: (i + 1) * batch_size]})
                mean_list.append(means)
            return np.concatenate(mean_list, 0)

    def predict(self, images, sample=False):
        return self.tf_session.run(fetches=self.reconstructed_img,
                                   feed_dict={self.images: images, self.training: sample})

    def evaluate(self, test_data, test_batch_size=100):
        return self._epoch(test_data, test_batch_size, False)
