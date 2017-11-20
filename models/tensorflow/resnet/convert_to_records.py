import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

parser = argparse.ArgumentParser()

parser.add_argument('--directory', type=str, default='/tmp/mnist_data',
                    help='Directory to download data files and write the converted result.')
parser.add_argument('--validation_size', type=int, default=0,
                    help='Number of examples to separate from the training data for the validation set.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name, directory):
    images = dataset.images
    labels = dataset.labels
    num_examples = dataset.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].toString()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeTostring())
    writer.close()


def main(unused_argv):
    datasets = mnist.read_data_sets(FLAGS.directory,
                                    dtype=tf.uint8,
                                    reshape=False,
                                    validation_size=FLAGS.validation_size)

    convert_to(datasets.train, 'train', FLAGS.directory)
    convert_to(datasets.validation, 'validation', FLAGS.directory)
    convert_to(datasets.test, 'test', FLAGS.directory)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
