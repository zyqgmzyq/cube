import os
import tensorflow as tf
from PIL import Image
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# train_dir = "D:\Yaoer\learn\Color_constancy\dataset\Cube\JPG\"
train_dir = "./image/"
# ground_truth = "D:\Yaoer\learn\Color_constancy\dataset\Cube\cube_gt.txt"
ground_truth = "./list.txt"
writer = tf.python_io.TFRecordWriter("train.tfrecords")
labels = np.zeros(3, dtype=np.float32)

for file, line in zip(os.listdir(train_dir), open(ground_truth)):
    print(line)
    ll = line.split(',')
    labels[0] = ll[0]
    labels[1] = ll[1]
    labels[2] = ll[2]
    print(labels)
    if file.endswith(".jpg"):
        file_path = train_dir + file
        print(file_path)
        img = Image.open(file_path)
        # 处理图片的大小
        img = img.resize((227, 227))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            # 图片对应单个结果
            # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[sort(file[:file.rindex("_")])])),
            # 图片对应多个结果
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())


writer.close()


# 读取tf
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 单结果的 label 返回是int
                                           # 'label': tf.FixedLenFeature([], tf.int64),
                                           # 数组返回， [3] 输入的数组的长度一样
                                           'label': tf.FixedLenFeature([3], tf.float32),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])

    # normalize
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = features['label']
    return img, label


img, label = read_and_decode("train.tfrecords")
img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=1, capacity=2000, min_after_dequeue=1000)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    for i in range(1):
        val, l = sess.run([img_batch, label_batch])
        print(l)
    print("complete ...")
    coord.request_stop()
    coord.join(threads)
    sess.close()

