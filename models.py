#encoding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

def conv(x, num_channel, kernel_size, stride=1, padding='SAME'):
    net = slim.conv2d(x, num_channel, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=None)
    net = layers.batch_norm(net, is_training=False, trainable=False)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, 0.5)
    return net

def deconv(x, num_channel, kernel_size, stride=2, padding='SAME'):
    net = slim.conv2d_transpose(x, num_outputs=num_channel, kernel_size=kernel_size, stride=stride,
                                padding=padding, activation_fn=None)
    net = layers.batch_norm(net, is_training=False, trainable=False)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, 0.5)
    return net

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

def Classifier(input):
    '''
    :param input: images with 224*224*3 size
    :return: vector of a feature layer with 1024 dimensions and possibility of seal or noseal picture
    '''
    with tf.variable_scope('Classifier'):
        with tf.variable_scope('conv_1'):
            net = conv(input, 64, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_2'):
            net = conv(net, 128, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_3'):
            net = conv(net, 256, [3, 3])
            net = conv(net, 256, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_4'):
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2)

        with tf.variable_scope('conv_5'):
            net = conv(net, 512, [3, 3])
            net = conv(net, 512, [3, 3])
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

def Discriminative(input):
    '''
    :param input:images with 224*224*3 size
    :return: vector of a feature layer with 1024 dimnesions and possibility of real picture
    '''
    with tf.variable_scope('Discriminative'):
        with tf.variable_scope('conv_1'):
            net = conv(input, 8, [3, 3], stride=2)

        with tf.variable_scope('conv_2'):
            net = conv(net, 16, [3, 3], stride=2)

        with tf.variable_scope('conv_3'):
            net = conv(net, 32, [3, 3], stride=2)

        with tf.variable_scope('conv_4'):
            net = conv(net, 64, [3, 3], stride=2)

        with tf.variable_scope('FC_1'):
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = layers.batch_norm(net, is_training=False, trainable=False)
            feature = tf.nn.relu(net)
            net = tf.nn.dropout(feature)

        with tf.variable_scope('FC_2'):
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.tanh)

    return feature, net