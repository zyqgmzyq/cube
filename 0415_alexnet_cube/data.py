import os
import tensorflow as tf
from PIL import Image
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = "/home/mvl/Dataset/ColorConstancy/Cube/JPG/"
label_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt.txt"
# img_height = 3456 
# img_width = 5184
img_height = 500
img_width = 1000
tf_file1 = "train1.tfrecords"
tf_file2 = "train2.tfrecords"
tf_file3 = "train3.tfrecords"
train1_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg1/"
train2_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg2/"
train3_dir = "/home/mvl/Dataset/ColorConstancy/Cube/jpg3/"
label1_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt1.txt"
label2_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt2.txt"
label3_dir = "/home/mvl/Dataset/ColorConstancy/Cube/cube_gt3.txt"


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


def write(filename, train_name, label_name):
    writer = tf.python_io.TFRecordWriter(filename)
    labels = np.zeros(3, dtype=np.float32)
    l = os.listdir(train_name)
    # 排序
    l = sorted(l, key=lambda name: int(name[:-4]))

    for file, line in zip(l, open(label_name)):
        ll = line.split(' ')
        labels[0] = ll[0]
        labels[1] = ll[1]
        labels[2] = ll[2]
        print(labels)
        if file.endswith(".jpg"):
            file_path = train_name + file
            print(file_path)
            img = Image.open(file_path)
            # 处理图片的大小
            img = img.resize((img_width, img_height))
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
def read_and_decode(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
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
    img = tf.reshape(img, [img_height, img_width, 3])
    # beishu = np.random.uniform(0.1, 1.0)
    # img = tf.random_crop(img, [int(beishu*img_height), int(beishu*img_height), 3])
    # angle = np.random.uniform(low=-30.0, high=30.0)
    # img = tf.contrib.image.rotate(img, angle)
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.resize_images(img, (int(img_height * 0.2), int(img_width * 0.2)))
    # img = tf.image.random_flip_left_right(img)
    # img = tf.random_crop(img, [448, 448, 3])
    
    # normalize
    img = tf.cast(img, tf.float32) * (1. / 255)
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    label = features['label']
    
    label = tf.nn.l2_normalize(label)
    return img, label


def get_batch(image, label, batch_size, capacity):
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=8,
                                                      capacity=capacity, min_after_dequeue=batch_size*10)
    return image_batch, label_batch


def test_read_and_decode(filenames):
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
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
    img = tf.reshape(img, [img_height, img_width, 3])
    # img = tf.image.resize_images(img, (520, 520))
    
    # normalize
    img = tf.cast(img, tf.float32) * (1. / 255)
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    label = features['label']
    
    label = tf.nn.l2_normalize(label)
    return img, label


def test_get_batch(image, label, batch_size, capacity):
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              capacity=capacity, num_threads=8)
    return image_batch, label_batch
    

if __name__ == '__main__':
    write(tf_file1, train1_dir, label1_dir)
    write(tf_file2, train2_dir, label2_dir)
    write(tf_file3, train3_dir, label3_dir)





