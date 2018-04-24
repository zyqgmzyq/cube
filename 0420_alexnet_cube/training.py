import tensorflow as tf
import data
import model
import os
from load_pytorch_weights import load_parameters


BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.00001

logs_train_dir1 = './logs/train12/'
logs_train_dir2 = './logs/train13/'
logs_train_dir3 = './logs/train23/'

tf_file1 = ["train1.tfrecords", "train2.tfrecords"]
tf_file2 = ["train1.tfrecords", "train3.tfrecords"]
tf_file3 = ["train2.tfrecords", "train3.tfrecords"]


def train(tf_file, logs_train_dir):
    filenames = tf.placeholder(tf.string, shape=[None])
    training_filenames = tf_file
    with tf.device('/cpu:0'):
        iterator = data.read_and_decode(filenames, BATCH_SIZE, True)
    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    tra_img, tra_label = iterator.get_next()
    train_logits, train_weight = model.inference(tra_img)
    train_loss = model.losses(train_logits, tra_label)
    train_op = model.trainning(train_loss, learning_rate)
    load_parameters("Alexnet.npy", sess, 'Alexnet')
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

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
    # train(tf_file1, logs_train_dir1)
    # train(tf_file2, logs_train_dir2)
    train(tf_file3, logs_train_dir3)


