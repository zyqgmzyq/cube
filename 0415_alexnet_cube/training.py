import tensorflow as tf
import data
import model
import os
import cv2 as cv
import matplotlib.pyplot as plt
from load_pytorch_weights import load_parameters


BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.000001

logs_train_dir = './logs/train12/'

tf_file1 = "train1.tfrecords"
tf_file2 = "train2.tfrecords"
tf_file3 = "train3.tfrecords"


def train():
    with tf.device('/cpu:0'):
        img, label = data.read_and_decode([tf_file1, tf_file2])
        img_batch, label_batch = data.get_batch(img, label, BATCH_SIZE, CAPACITY)
    train_logits, train_weight = model.inference(img_batch)
    # s = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ALexNet')
    # print(s)
    # for v in s:
    #     print(v.name)
    # exit(0)

    train_loss = model.losses(train_logits, label_batch)
    train_op = model.trainning(train_loss, learning_rate)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_parameters("Alexnet.npy", sess, 'Alexnet')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss = sess.run([train_op, train_loss])
                
                if step % 50 == 0:
                    # tra_weight.imshow()
                    print('Step %d, train loss = %.2f, l2 loss = %.2f' % (step, tra_loss, tra_loss))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    train()


