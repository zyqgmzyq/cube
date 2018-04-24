import numpy as np
import tensorflow as tf


def load_parameters(filename, sess, scope='Alexnet'):
    """load .npy pytorch weights

    filname: weights file in .npy format
    sess: current session
    scope: scope of weights
    """
    layers = np.load(filename, encoding = 'latin1')
    c = 0
    tensors = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    for layer in layers:
        for i, parameter in enumerate(layer):
            if len(parameter.shape) == 4:
                parameter = np.transpose(parameter, [2, 3, 1, 0])
            elif len(parameter.shape) == 2:
                parameter = np.transpose(parameter, [1, 0])
            print(tensors[c], parameter.shape)
            sess.run(tf.assign(tensors[c], parameter))
            print(tensors[c], " load successfully")
            c += 1

if __name__ == '__main__':
    phase = tf.placeholder(tf.bool, name='phase')
    sess = tf.Session()
    logits = ResNet.ResNet_v1(
        tf.placeholder(tf.float32, [32, 224, 224, 3]),
        True, [2, 2, 2, 2]
    )
    load_parameters('ResNet.npy', sess)

