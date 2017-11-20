import argparse
import os
import sys

import tensorflow as tf

import resnet_model
import preprocessing

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='',
                    help='The directory where the ImageNet input data is stored.')
parser.add_argument('--model_dir', type=str, default='/tmp/resnet_model',
                    help='The directory where the model will be stored.')
parser.add_argument('--resnet_size', type=int, default=50, choices=[18, 34, 50, 101, 152, 200],
                    help='The size of the ResNet model to use.')
parser.add_argument('--train_epochs', type=int, default=100,
                    help='The number of epochs to use for training.')
parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training and evaluation.')
parser.add_argument('--data_format', type=str, default=None, choices=['channels_first', 'channels_last'],
                    help='A flag to override the data format used in the model. channels_first '
                         'provides a performance boost on GPU but is not always compatible '
                         'with CPU. If left unspecified, the data format will be chosen '
                         'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_LABEL_CLASSED = 1001

_MOMENTUN = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500


def filename(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, 'train-%05d-of-01024' % i) for i in range(1024)]
    else:
        return [os.path.join(data_dir, 'validation-%05d-of-0128' % i) for i in range(128)]


def record_parser(value, is_training):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/bbox/xmin': tf.FixedLenFeature((), tf.float32),
        'image/object/bbox/ymin': tf.FixedLenFeature((), tf.float32),
        'image/object/bbox/xmax': tf.FixedLenFeature((), tf.float32),
        'image/object/bbox/ymax': tf.FixedLenFeature((), tf.float32),
        'image/object/class/label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(value, keys_to_features)
    image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = preprocessing.preprocess_image(image=image,
                                           output_height=_DEFAULT_IMAGE_SIZE,
                                           output_width=_DEFAULT_IMAGE_SIZE,
                                           is_training=is_training)
    label = tf.cast(tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)
    return image, tf.one_hot(label, _LABEL_CLASSED)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    dataset = tf.data.Dataset.from_tensor_slices(filename(is_training, data_dir))
    if is_training:
        dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value, is_training), num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)

    if is_training:
        # larger sizes result in better randomness, while smaller sizes hae better performance
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # Call repeat after shuffling rather then before, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.next()
    return images, labels


def resnet_model_fn(features, labels, mode, params):
    network = resnet_model.imagenet_resnet_v2(params['resnet_size'], _LABEL_CLASSED, params['data_format'])
    logits = network(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size.
        # when the batch size is 256, then learning rate should be 0.1
        initial_leaning_rate = 0.1 * params['batch_size'] / 256
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        boundaries = [int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        values = [initial_leaning_rate * decay for decay in [1, 0.1, 00.1, 1e-3, 1e-4]]
        leaning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)

        tf.identity(leaning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', leaning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=leaning_rate, momentum=_MOMENTUN)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.convert_to_tensor(update_ops):
            train_op = optimizer.minimize(logits, global_step)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name="train_accuracy")
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    run_config = tf.estimator.RunConfig.replace(save_checkpoints_sex=1e9)
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn,
        model_dir=FLAGS.mode_dir,
        config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tenfors_to_log = {
            'learning_rate': 'leanring_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }
        logging_hook = tf.train.LoggingTensorHook(tensors=tenfors_to_log, every_n_iter=100)

        print('Starting a training cycle.')
        resnet_classifier.train(
            input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        print('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
