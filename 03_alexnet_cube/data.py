import os
import tensorflow as tf
from sklearn.model_selection import KFold
from PIL import Image
import numpy as np
import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = "/home/mvl/Dataset/ColorConstancy/Cube/JPG/"
# train_dir = "./image/"
ground_truth = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt.txt"
# ground_truth = "./list.txt"


def normalize(image, means, stds):
    """normalize a image by substract mean and std
    Args :
        image: 3-D image
        means: list of C mean values
        stds: list of C std values
    Returns:
        3-D normalized image
        
    for imagenet pretrained model:
        means = [0.485, 0.456, 0.406] R G B
        std = [0.229, 0.224, 0.225] R G B
    """

    num_channels = image.get_shape().as_list()[-1]
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
        channels[i] /= stds[i]
    return tf.concat(axis=2, values=channels)


def write():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    labels = np.zeros(3, dtype=np.float32)

    l = os.listdir(train_dir)
    l = sorted(l, key=lambda name: int(name[:-4]))
    for file, line in zip(l, open(ground_truth)):
        ll = line.split(' ')
        labels[0] = ll[0]
        labels[1] = ll[1]
        labels[2] = ll[2]
        print(labels)
        if file.endswith(".jpg"):
            file_path = train_dir + file
            print(file_path)
            img = Image.open(file_path)
            # 处理图片的大小
            img = img.resize((224, 224))
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
    img = tf.reshape(img, [224, 224, 3])

    # normalize
    img = tf.cast(img, tf.float32) * (1. / 255)
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    label = features['label']
    label = tf.nn.l2_normalize(label)

    return img, label


def get_batch(image, label, batch_size, capacity):
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=1000)
    return image_batch, label_batch


if __name__ == '__main__':
    write()





