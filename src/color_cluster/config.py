import tensorflow as tf

NORMALIZATION_FN_CONV = None #lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
NORMALIZATION_FN_DENSE = None
ACTIVATION_FN = tf.nn.relu


KERNEL_INITIALIZER = tf.contrib.layers.xavier_initializer()
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
BIAS_INITIALIZER = tf.zeros_initializer()

default_filter_size = 3
default_num_filters = 12

'''Pooling '''
POOL_FN = tf.layers.max_pooling2d
default_pool_size = 3
default_pool_stride = 2
default_conv_stride = 1

'''NETWORK'''
PADDING = 'VALID'
'''Convolutional Layer 1'''
CONV1_FILTER_SIZE = default_filter_size
CONV1_NUM_FILTERS = default_num_filters
CONV1_STRIDE = default_conv_stride
POOL1_FILTER_SIZE = default_pool_size
POOL1_STRIDE = default_pool_stride

'''Convolutional Layer 2'''
CONV2_FILTER_SIZE = default_filter_size
CONV2_NUM_FILTERS = default_num_filters
CONV2_STRIDE = default_conv_stride
POOL2_FILTER_SIZE = default_pool_size
POOL2_STRIDE = default_pool_stride

'''Convolutional Layer 3'''
CONV3_FILTER_SIZE = default_filter_size
CONV3_NUM_FILTERS = default_num_filters
CONV3_STRIDE = default_conv_stride

POOL3_FILTER_SIZE = default_pool_size
POOL3_STRIDE = default_pool_stride

'''Dense Layer 4'''
DENSE4_NUM_OUTPUTS = 30

'''Dense Layer 5'''
LATENT_DIMENSION = 3