import tensorflow as tf
import data
import model
import os
import cv2 as cv
import numpy as np
from load_pytorch_weights import load_parameters

BATCH_SIZE = 1
CAPACITY = 2000
MAX_STEP = 455

logs_test_dir3 = './logs/test3/'
logs_test_dir2 = './logs/test2/'
logs_test_dir1 = './logs/test1/'

loss_dir3 = "./loss3.txt"
loss_dir2 = "./loss2.txt"
loss_dir1 = "./loss1.txt"

restore_dir3 = './logs/train12/model.ckpt-9999'
restore_dir2 = './logs/train13/model.ckpt-9999'
restore_dir1 = './logs/train23/model.ckpt-9999'

weight_write_dir3 = "/home/mvl/Dataset/ColorConstancy/Cube/test3_v/weight{}.jpg"
weight_write_dir2 = "/home/mvl/Dataset/ColorConstancy/Cube/test2_v/weight{}.jpg"
weight_write_dir1 = "/home/mvl/Dataset/ColorConstancy/Cube/test1_v/weight{}.jpg"

tf_file1 = ["train1.tfrecords"]
tf_file2 = ["train2.tfrecords"]
tf_file3 = ["train3.tfrecords"]
list_loss = []


def evaluate(tf_file, logs_test_dir, restore_dir, loss_dir, weight_write_dir):
    filenames = tf.placeholder(tf.string, shape=[None])
    validation_filenames = tf_file
    iterator = data.read_and_decode(filenames, BATCH_SIZE, False)

    sess = tf.Session()
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
    test_img, test_label = iterator.get_next()
    test_logits, test_weight = model.inference(test_img)
    test_loss = model.losses(test_logits, test_label)
    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
    saver = tf.train.Saver()
    saver.restore(sess, restore_dir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            test_logits_val, test_loss_val, test_label_val = sess.run([test_logits, test_loss, test_label])

            print('label:', test_label_val)
            print('estimate:', test_logits_val)
            print('Step %d, test loss = %.2f' % (step + 1, test_loss_val))
            list_loss.append(test_loss_val)
            with open(loss_dir, 'a') as f:
                f.write('%.6f' % (test_loss_val))
                f.write("\n")

            tra_weight = sess.run(test_weight)
            print(tra_weight.shape)
            for j in range(tra_weight.shape[0]):
                tra_weight_draw = tra_weight[j, :, :, :]
                tra_weight_draw = (tra_weight_draw - tra_weight_draw.min()) / \
                                  (tra_weight_draw.max() - tra_weight_draw.min())
                tra_weight_draw *= 255.0
                tra_weight_draw = cv.resize(tra_weight_draw, (600, 400))
                cv.imwrite(weight_write_dir.format(step),
                           tra_weight_draw.astype(np.uint8))

            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_test_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        print('mean', sess.run(tf.reduce_sum(list_loss) / MAX_STEP))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    cv.waitKey(0)


if __name__ == '__main__':
    # evaluate(tf_file3, logs_test_dir3, restore_dir3, loss_dir3, weight_write_dir3)
    # evaluate(tf_file2, logs_test_dir2, restore_dir2, loss_dir2, weight_write_dir2)
   
    evaluate(tf_file1, logs_test_dir1, restore_dir1, loss_dir1, weight_write_dir1)


