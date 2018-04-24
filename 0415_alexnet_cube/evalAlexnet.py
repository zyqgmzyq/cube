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

logs_test_dir = './logs/test3/'

tf_file1 = ["train1.tfrecords"]
tf_file2 = ["train2.tfrecords"]
tf_file3 = ["train3.tfrecords"]
list_loss = []


def evaluate():
    test_img, test_label = data.test_read_and_decode([tf_file3])
    test_img_batch, test_label_batch = data.test_get_batch(test_img, test_label, BATCH_SIZE, CAPACITY)
    test_logits, test_weight = model.inference(test_img_batch)
    test_loss = model.losses(test_logits, test_label_batch)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # load_parameters("Alexnet.npy", sess, 'Alexnet')
        train_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)
        saver = tf.train.Saver()
        saver.restore(sess, './logs/train12/model.ckpt-9999')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                test_logits_val, test_loss_val, test_label_val = sess.run([test_logits, test_loss, test_label])
                
                print('label:', test_label_val)
                print('estimate:', test_logits_val)
                print('Step %d, test loss = %.2f' % (step+1, test_loss_val))
                list_loss.append(test_loss_val)
                with open("./color31.txt", 'a') as f:
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
                    cv.imwrite("/home/mvl/Dataset/ColorConstancy/Cube/test3_v/weight{}.jpg".format(step),
                               tra_weight_draw.astype(np.uint8))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_test_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            print('mean', sess.run(tf.reduce_sum(list_loss)/MAX_STEP))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        cv.waitKey(0)        


if __name__ == '__main__':
    evaluate()


