import tensorflow as tf
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dropout = 0.5
format_data = 'channels_last'


# 定义显示网络结构的函数，展示输出tensor的尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    with tf.variable_scope('Alexnet') as scope:
        # conv1
        with tf.variable_scope('conv1') as scope:
            conv = tf.layers.conv2d(images, filters=64, kernel_size=[11, 11],
                                    strides=[4, 4], padding='same', data_format=format_data)
            conv1 = tf.nn.relu(conv)
            print_activations(conv1)

        # pool1 and norm1
        with tf.variable_scope('pooling1_lrn') as scope:
            # norm1 = tf.nn.lrn(conv1, depth_radius=5, bias=1.0, alpha=0.001 / 9.0,
            #                  beta=0.75, name='norm1',)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=[2, 2],
                                            padding='valid', data_format=format_data)
            print_activations(pool1)

        # conv2
        with tf.variable_scope('conv2') as scope:
            conv = tf.layers.conv2d(pool1, filters=192, kernel_size=[5, 5], strides=[1, 1],
                                    padding='same', data_format='channels_last')
            conv2 = tf.nn.relu(conv)
            print_activations(conv2)

        # pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            # norm2 = tf.nn.lrn(conv2, depth_radius=5, bias=1.0, alpha=0.001 / 9.0,
            #                 beta=0.75, name='norm1')
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=[2, 2],
                                            padding='valid', data_format=format_data)
            print_activations(pool2)
        # conv3
        with tf.variable_scope('conv3') as scope:
            conv = tf.layers.conv2d(pool2, filters=384, kernel_size=[3, 3], strides=[1, 1],
                                    padding='same', data_format=format_data)
            conv3 = tf.nn.relu(conv)
            print_activations(conv3)

        # conv4
        with tf.variable_scope('conv4') as scope:
            conv = tf.layers.conv2d(conv3, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                    padding='same', data_format=format_data)
            conv4 = tf.nn.relu(conv)
            print_activations(conv4)

        # conv5
        with tf.variable_scope('conv5') as scope:
            conv = tf.layers.conv2d(conv4, filters=256, kernel_size=[3, 3], strides=[1, 1],
                                    padding='same', data_format=format_data)
            conv5 = tf.nn.relu(conv)
            print_activations(conv5)

        # pooling5
        with tf.variable_scope('pool5') as scope:
            pool5 = tf.layers.average_pooling2d(conv5, pool_size=[3, 3], strides=[2, 2],
                                                padding='valid', data_format=format_data)
            print_activations(pool5)


    # conv6
    with tf.variable_scope('conv6') as scope:
        conv = tf.layers.conv2d(pool5, filters=64, kernel_size=[6, 6], strides=[1, 1],
                                padding='same', data_format=format_data)
        conv6 = tf.nn.relu(conv)
        conv6 = tf.nn.dropout(conv6, dropout)
        print_activations(conv6)

    # conv7
    with tf.variable_scope('conv7') as scope:
        conv = tf.layers.conv2d(conv6, filters=4, kernel_size=[1, 1], strides=[1, 1],
                                padding='same', data_format=format_data)
        conv7 = tf.nn.relu(conv)
        print_activations(conv7)

    #  features maps
    red, green, blue, weight = tf.split(conv7, 4, axis=3)

    # rgb
    r = tf.multiply(red, weight)
    print_activations(r)
    g = tf.multiply(green, weight)
    b = tf.multiply(blue, weight)
    rgb = tf.concat([r, g, b], axis=3)
    print_activations(rgb)

    # summation
    sum_rgb = tf.reduce_sum(rgb, axis=[1, 2])
    print_activations(sum_rgb)

    # normalization
    color = tf.nn.l2_normalize(sum_rgb, axis=1)
    print_activations(color)

    return color


# %%loss
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        loss = 180.0 * tf.losses.cosine_distance(labels=labels, predictions=logits, axis=1)
        # loss = tf.multiply(180/math.pi, tf.acos(tf.reduce_sum(logits * labels, axis=-1)))
        loss2 = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        loss = tf.reduce_mean(loss, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
        # tf.summary.scalar(scope.name + '/loss', loss2)
    return loss, loss2


# %%
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(loss):

    return loss


