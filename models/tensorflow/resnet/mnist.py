import argparse
import os
import sys

import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100, help='Number of images to process in a batch')
parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data', help='Path to the MNIST data directory.')
parser.add_argument('--model_dir', type=str, default='/tmp/mnist_model',
                    help='The directory where the model will be stored.')
parser.add_argument('--train_epochs', type=int, default=40, help='Number of epochs to train.')
parser.add_argument(
    '--data_format', type=str, default=None, choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000
}


def input_fn(is_training, filename, batch_size=1, num_epochs=1):
    def example_parser(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([28 * 28])

        image = tf.cast(image, tf.float32) / 255 - 0.5
        label = tf.cast(features['label'], tf.int32)
        return image, tf.one_hot(label, 10)

    dataset = tf.data.TFRecordDataset([filename])

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(example_parser).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def mnist_model(inputs, mode, data_format):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             data_format=data_format)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2,
                                    data_format=data_format)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             data_format=data_format)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    data_format=data_format)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits


def mnist_model_fn(features, labels, mode, params):
    logits = mnist_model(features, mode, params['data_format'])
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )


def main(unused_argv):
    train_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    test_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    assert (tf.gfile.Exists(train_file) and tf.gfile.Exists(test_file)), (
        'Run covert_to_records.py to convert the MNIST data to TFRecord file format.')

    mnist_classfier = tf.estimator.Estimator(model_fn=mnist_model_fn, model_dir=FLAGS.mode_dir,
                                             params={'data_format': FLAGS.data_format})

    tensors_to_log = {
        'train_accuracy': 'train_accuracy'
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    mnist_classfier.train(input_fn=lambda: input_fn(True, train_file, FLAGS.batch_size, FLAGS.train_epochs),
                          hooks=[logging_hook])

    eval_results = mnist_classfier.evaluate(input_fn=lambda: input_fn(False, test_file, FLAGS.batch_size))
    print()
    print('Evaluation results:\n\t%s' % eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
