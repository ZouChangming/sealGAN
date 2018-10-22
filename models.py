#encoding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

def Leaky_Relu(x):
    return tf.maximum(x, 0.2*x)

def conv(x, num_channel, kernel_size, stride=1, keep_prob=1.0, padding='SAME', activation='relu'):
    net = slim.conv2d(x, num_channel, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=None)
    net = layers.batch_norm(net, is_training=False, trainable=False)
    if activation == 'Leaky_Relu':
        net = Leaky_Relu(net)
    else:
        net = tf.nn.relu(net)
    net = tf.nn.dropout(net, keep_prob)
    return net

def deconv(x, num_channel, kernel_size, stride=2, padding='SAME', activation='relu'):
    net = slim.conv2d_transpose(x, num_outputs=num_channel, kernel_size=kernel_size, stride=stride,
                                padding=padding, activation_fn=None)
    net = layers.batch_norm(net, is_training=False, trainable=False)
    if activation == 'Leaky_Relu':
        net = Leaky_Relu(net)
    else:
        net = tf.nn.relu(net)
    # net = tf.nn.dropout(net, 0.8)
    return net

def residule_block(x, dim, ks=3, s=1):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = conv(y, dim, [ks, ks], stride=s, padding='VALID')
    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = conv(y, dim, [ks, ks], stride=s, padding='VALID')
    return y + x

def Encoder(input):
    '''
    :param input:original images with size (224, 224, 3)
    :return: feature vector with 256 dimensions
    '''
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('conv_1'):
            net = conv(input, 64, [3, 3])
            net = conv(net, 64, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [3, 3])
            net = conv(net, 128, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [3, 3])
            net = conv(net, 256, [3, 3])
            net = conv(net, 256, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_4'):
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_5'):
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('FC_1'):
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4069, activation_fn=None)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)

        with tf.variable_scope('FC_2'):
            net = slim.fully_connected(net, 4096, activation_fn=None)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)

        with tf.variable_scope('FC_3'):
            net = slim.fully_connected(net, 256)

    return net

def Classifier(input, reuse=False):
    '''
    :param input: images with 224*224*3 size
    :return: vector of a feature layer with 1024 dimensions and possibility of seal or noseal picture
    '''
    with tf.variable_scope('Classifier', reuse=reuse):
        with tf.variable_scope('conv_1'):
            net = conv(input, 64, [3, 3], keep_prob=0.8)
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [3, 3], keep_prob=0.8)
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [3, 3], keep_prob=0.8)
            net = conv(net, 256, [3, 3], keep_prob=0.8)
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_4'):
            net = conv(net, 512, [3, 3], keep_prob=0.8)
            net = conv(net, 512, [3, 3], keep_prob=0.8)
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_5'):
            net = conv(net, 512, [3, 3], keep_prob=0.8)
            net = conv(net, 512, [3, 3], keep_prob=0.8)
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('FC_1'):
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, activation_fn=None)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)

        with tf.variable_scope('FC_2'):
            net = slim.fully_connected(net, 1024, activation_fn=None)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            feature = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)

        with tf.variable_scope('FC_3'):
            net = slim.fully_connected(net, 2, activation_fn=None)

    return feature, net

def Generative(input):
    '''
    :param input: a vector with 256 dimensions
    :return: images with 224*224*3 size
    '''
    with tf.variable_scope('Generative'):
        with tf.variable_scope('reshape'):
            net = slim.fully_connected(input, 7*7*256, activation_fn=None)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.5)
            net = tf.reshape(net, [-1, 7, 7, 256])

        with tf.variable_scope('conv_1'):
            net = deconv(net, 256, [3, 3])

        with tf.variable_scope('conv_2'):
            net = deconv(net, 256, [3, 3])

        with tf.variable_scope('conv_3'):
            net = deconv(net, 128, [5, 5])

        with tf.variable_scope('conv_4'):
            net = deconv(net, 92, [5, 5])

        with tf.variable_scope('conv_5'):
            net = deconv(net, 64, [5, 5])

        with tf.variable_scope('conv_6'):
            net = slim.conv2d(net, 3, kernel_size=[5, 5], stride=1, padding='SAME', activation_fn=tf.nn.tanh)

    return net

def Generative2(input):
    '''
    :param input:
    :return:
    '''
    with tf.variable_scope('Generative'):
        with tf.variable_scope('conv_1'):
            net = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            net = conv(net, 64, [7, 7], padding='VALID')

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [3, 3], stride=2)

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [3, 3], stride=2)

        with tf.variable_scope('refine'):
            with tf.variable_scope('res_block_1'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_2'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_3'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_4'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_5'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_6'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_7'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_8'):
                net = residule_block(net, 256)
            with tf.variable_scope('res_block_9'):
                net = residule_block(net, 256)

        with tf.variable_scope('deconv_1'):
            net = deconv(net, 128, [3, 3])

        with tf.variable_scope('deconv_2'):
            net = deconv(net, 64, [3, 3])

        with tf.variable_scope('conv_4'):
            net = tf.pad(net, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            net = slim.conv2d(net, 3, kernel_size=[7, 7], stride=1, padding='VALID', activation_fn=tf.nn.tanh)

    return net


def Discriminative(input, reuse=False):
    '''
    :param input:images with 224*224*3 size
    :return: possibility of real picture
    '''
    with tf.variable_scope('Discriminative', reuse=reuse):
        with tf.variable_scope('conv_1'):
            net = slim.conv2d(input, 64, kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=None)
            net = Leaky_Relu(net)

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_4'):
            net = conv(net, 512, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_5'):
            net = slim.conv2d(net, 1, kernel_size=[4, 4], stride=1, padding='SAME', activation_fn=None)

    return net

def Discriminative2(input, reuse=False):
    '''
        :param input:images with 224*224*3 size
        :return: possibility of real picture
        '''
    with tf.variable_scope('Discriminative', reuse=reuse):
        with tf.variable_scope('conv_1'):
            net = slim.conv2d(input, 64, kernel_size=[4, 4], stride=2, padding='SAME', activation_fn=None)
            net = Leaky_Relu(net)

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_4'):
            net = conv(net, 512, [4, 4], stride=2, activation='Leaky_Relu')

        with tf.variable_scope('conv_5'):
            net = conv(net, 256, [4, 4], stride=1, activation='Leaky_Relu')

        with tf.variable_scope('conv_6'):
            net = conv(net, 64, [4, 4], stride=1, activation='Leaky_Relu')

        with tf.variable_scope('conv_7'):
            net = slim.conv2d(net, 1, kernel_size=[4, 4], stride=1, padding='SAME', activation_fn=None)

    return net