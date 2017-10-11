import time
import tensorflow as tf
import numpy as np
import scipy.misc as sp
import scipy.ndimage.interpolation as inter
import scipy.ndimage as ndimage
import cv2

class FCN:
    """
    A FCN that is based on the VGG16 weights, the weights are constants
    so they dont train and just there to provide the features

    """
    #
    # Initialize network
    #

    def __init__(self, input_size, nb_class, weight_path, nb_kernels = 16, kernel_size=3, padding='SAME', pos_weight=10, downscale_factor = 2):
        # Keep the data and settings
        self.input_size = input_size
        self.nb_class = nb_class
        self.data_dict = np.load(weight_path, encoding='latin1').item()

        # Create network
        tf.reset_default_graph()
        self._build_graph(input_size, nb_class, nb_kernels, kernel_size, padding, pos_weight=pos_weight)
        self._initialize_session()
        self.downscale_factor = downscale_factor

    def _build_graph(self, input_size, nb_class, nb_kernels, kernel_size, padding, pos_weight):

        # Inputs of the network
        with tf.name_scope('inputs'):
            self.image = tf.placeholder(dtype=tf.float32, shape=[None, input_size[0], input_size[1], 3], name='input_image')
            self.label = tf.placeholder(dtype=tf.float32, shape=[None, nb_class], name='labels')
            self.drop = tf.placeholder(dtype=tf.float32, shape=[1], name='dropout_rate')

        # Convolutional part
        VGG_MEAN = [103.939, 116.779, 123.68]

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.image)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")

        self.conv6_1 = self._create_conv(self.conv4_3, nb_kernels, kernel_size, padding, "conv6_1")
        self.conv6_2 = self._create_conv(self.conv6_1, nb_kernels, kernel_size, padding, "conv6_2")
        self.hmap = self._create_conv(self.conv6_2, self.nb_class, kernel_size, padding, "hmap")
        GAP = self._create_glb_pool(self.hmap, "GAP")
        self.output = tf.nn.tanh(GAP, "output")

        # Metrics
        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(tf.round(self.output), tf.round(self.label))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Loss function
        self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=GAP, pos_weight=pos_weight, name='loss')

        # Build the trainstep, if we want to we can add a regularizer loss
        optimizer = tf.train.AdamOptimizer()
        self.trainstep = optimizer.minimize(self.loss, name='train_step')

        # Remove the data_dict
        del self.data_dict

    def _initialize_session(self):
        """
        Initialize session, variables, saver
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

    #
    # Building blocks of Network
    #

    def _create_drop_out(self, in_tensor, rate, name):
        with tf.name_scope(name):
            return tf.layers.dropout(in_tensor, rate=rate, name=name)

    def _create_avg_pool(self, in_tensor, pool_size, strides, padding, name):
        with tf.name_scope(name):
            return tf.layers.average_pooling2d(in_tensor, pool_size=pool_size, strides=strides, padding=padding)

    def _create_max_pool(self, in_tensor, pool_size, strides, padding, name):
        with tf.name_scope(name):
            return tf.layers.max_pooling2d(in_tensor, pool_size=pool_size, strides=strides, padding=padding)

    def _create_conv(self, in_tensor, nb_kernels, kernel_size, padding, name):
        with tf.name_scope(name):
            return tf.layers.conv2d(in_tensor, filters=nb_kernels, kernel_size=kernel_size, activation=tf.nn.relu, padding=padding)

    def _create_glb_pool(self, in_tensor, name):
        with tf.name_scope(name):
            return tf.reduce_mean(in_tensor, [1, 2])


    #
    # Load weights
    #

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    #
    # Methods of the network
    #

    def train(self, trn_data, val_data, train_params):
        """
        Train our network, this function defines the
        general training. The _epoch function describes
        1 training epoch.

        :param trn_data: an array containing trainig data, img first, label second
        :param val_data: an array containing validation data, see ^^
        :param train_params:
        """

        # Settings
        drop = train_params['drop']
        nb_epoch = train_params['nb_epoch']
        batch_sz = train_params['batch_sz']
        save_dir = train_params['save_dir']
        best_scr = np.infty

        # Start running
        for epoch in range(1, nb_epoch + 1):
            print ('Epoch number: {}'.format(epoch))

            # Train and get the validation
            # trn_loss, trn_acc = self._epoch(trn_data, drop, batch_sz, train=True)
            # val_loss, val_acc = self._epoch(val_data, drop, batch_sz, train=False)
            # print('Training: loss = {}, acc = {}'.format(trn_loss, trn_acc))
            # print('Validation: loss = {}, acc = {}'.format(val_loss, val_acc))

            trn_loss, _ = self._epoch(trn_data, drop, batch_sz, train=True)
            val_loss, _ = self._epoch(val_data, drop, batch_sz, train=False)
            print('Training: loss = {}, acc = '.format(trn_loss))
            print('Validation: loss = {}, acc = '.format(val_loss))

            if val_loss < best_scr:
                print('Validation score has improved from {} to {}, saving the model\n'.format(best_scr, val_loss))
                best_scr = val_loss
                self.save(save_dir)
            else:
                print('Validation loss did not improve \n')

    def _epoch(self, data, drop, batch_sz, train):
        """

        :param data:
        :param batch_sz:
        :param train:
        """

        # Collectors for the loss and accuracy data
        loss_coll = np.empty(shape=(0,))
        acc_coll = np.empty(shape=(0,))

        # Get the data
        _img = data[0]
        _lbl = data[1]

        # Shuffle
        idx = np.random.permutation(list(range(_img.shape[0])))
        idx_chunks = [idx[i:i + batch_sz] for i in range(0, len(idx), batch_sz)]

        # Go over the data
        for _, chunk in enumerate(idx_chunks):
            # Get the feed_dict
            feed_dict = {
                self.image: _img[chunk],
                self.label: _lbl[chunk],
                self.drop: [drop]
            }

            # Run it through the graph
            if train:
                _, loss, accuracy = self._sess.run(
                    [self.trainstep, self.loss, self.accuracy],
                    feed_dict = feed_dict)

            else:
                loss, accuracy = self._sess.run(
                    [self.loss, self.accuracy],
                    feed_dict = feed_dict)

            loss = np.average(loss, axis=1)
            loss_coll = np.hstack((loss_coll, loss))
            # accuracy = np.average(accuracy, axis=1)
            # acc_coll = np.hstack((acc_coll, accuracy))

        return np.mean(loss_coll), np.mean(acc_coll)


    def load(self, path):
        saver = tf.train.import_meta_graph(path + '/.meta')
        saver.restore(self._sess, path)

    def save(self, path):
        self._saver.save(self._sess, path)

    def extract_hmap(self, data, have_brick=False):
        _image = sp.imresize(data, self.input_size)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        _label = np.zeros(shape=(_image.shape[0], self.nb_class))
        drop = 0


        feed_dict = {
            self.image: [_image],
            self.label: [_label[0]],
            self.drop: [drop]
        }
        output, heat_map, omap = self._sess.run([self.output, self.hmap, self.conv3_3], feed_dict=feed_dict)
	
        heat_map = np.squeeze(heat_map)
        omap = np.squeeze(omap)

        omap = np.mean(omap,axis=2)

 #       if np.average(output) > 0.5:
        thresholded_heat_map = sp.imresize(heat_map, self.downscale_factor * np.array(self.input_size))
            
        #    thresholded_heat_map[-6:, :] = 0
        #    thresholded_heat_map[:, :10] = 0
        #    thresholded_heat_map = np.roll(thresholded_heat_map, 6, axis=0)
        #    thresholded_heat_map = np.roll(thresholded_heat_map, -10, axis=1)            
            
        #    thresholded_heat_map[thresholded_heat_map < 50] = 0
        #    thresholded_heat_map = thresholded_heat_map.astype(np.uint8)
#        else:
         
#	    zeros_shape = self.downscale_factor * np.array(self.input_size)
#            thresholded_heat_map = np.zeros(shape=zeros_shape, dtype=np.uint8)
        
        if have_brick:
            thresholded_omap = sp.imresize(omap, self.downscale_factor * np.array(self.input_size))
            thresholded_omap[thresholded_omap < 50] = 0
            thresholded_omap = thresholded_omap.astype(np.uint8)
        else:
            # Threshold omap
            thresholded_omap = sp.imresize(omap, self.downscale_factor * np.array(self.input_size))
            thresholded_omap[sp.imresize(ndimage.filters.gaussian_filter(heat_map, 1.25), self.downscale_factor * np.array(self.input_size)) > 10] = 0
            thresholded_omap[thresholded_omap < 50] = 0
            thresholded_omap = thresholded_omap.astype(np.uint8)
             
        return thresholded_heat_map, thresholded_omap

