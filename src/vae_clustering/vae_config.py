import tensorflow as tf

NORMALIZATION_FN_CONV = None #lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
NORMALIZATION_FN_DENSE = None # lambda x: tf.contrib.layers.layer_norm(x, center=True, scale=True)
ACTIVATION_FN = tf.nn.relu


KERNEL_INITIALIZER = tf.contrib.layers.xavier_initializer()
WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()
BIAS_INITIALIZER = tf.zeros_initializer()

default_filter_size = 3
default_num_filters = 12


'''Pooling '''
POOL_FN = tf.contrib.layers.max_pool2d
default_pool_size = 2
default_pool_stride = 2


'''ENCODER'''
ENC_PADDING = "VALID"
'''Convolutional Layer 1'''
ENC_CONV1_FILTER_SIZE = default_filter_size
ENC_CONV1_NUM_FILTERS = default_num_filters
ENC_POOL1_FILTER_SIZE = default_pool_size
ENC_POOL1_STRIDE = default_pool_stride

'''Convolutional Layer 2'''
ENC_CONV2_FILTER_SIZE = default_filter_size
ENC_CONV2_NUM_FILTERS = default_num_filters
ENC_POOL2_FILTER_SIZE = default_pool_size
ENC_POOL2_STRIDE = default_pool_stride

'''Convolutional Layer 3'''
ENC_CONV3_FILTER_SIZE = default_filter_size
ENC_CONV3_NUM_FILTERS = default_num_filters
ENC_POOL3_FILTER_SIZE = default_pool_size
ENC_POOL3_STRIDE = default_pool_stride

'''Dense Layer 4'''
ENC_DENSE4_NUM_OUTPUTS = 100

'''DECODER'''
default_dec_stride = 2
INITIAL_IMAGE_SIZE = 8
INITIAL_NUM_CHANNELS = default_num_filters
DEC_PADDING = "SAME"
'''Transposed Convolutional Layer 2'''
DEC_CONV2_FILTER_SIZE = 4
DEC_CONV2_NUM_FILTERS = default_num_filters
DEC_CONV2_STRIDE = default_dec_stride

'''Transposed Convolutional Layer 3'''
DEC_CONV3_FILTER_SIZE = 4
DEC_CONV3_NUM_FILTERS = default_num_filters
DEC_CONV3_STRIDE = default_dec_stride

'''Transposedd Convolutional Layer 4'''
DEC_CONV4_FILTER_SIZE = 4
DEC_CONV4_NUM_FILTERS = default_num_filters
DEC_CONV4_STRIDE = default_dec_stride

'''Output'''
DEC_OUTPUT_FILTER_SIZE = 3