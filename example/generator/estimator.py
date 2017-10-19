import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tensorflow import train
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.slim import nets as slim_nets

tf.flags.DEFINE_string('name', 'train', 'used to decide mode_dir')
tf.flags.DEFINE_string('mode', 'train', 'one of [train, eval, infer, export]')
tf.flags.DEFINE_string('model', 'resnet_v2_50', 'one of [custom, inception_v3, resnet_v2_50]')
tf.flags.DEFINE_string('categories', 'clean,mark_image', "categories, splited by ','")
tf.flags.DEFINE_string('data_path', '', 'directory or file path of data')
tf.flags.DEFINE_string('infer_image_path', '', 'file path of image file or directory to infer')
tf.flags.DEFINE_integer('batch_size', '64', 'batch size')
tf.flags.DEFINE_integer('num_epochs', '5', 'epoch number')
tf.flags.DEFINE_float('lr', '0,001', 'learning rate')
flags = tf.flags.FLAGS

CATEGORIES = []


def mode_custom(inputs, training, num_classes):
    inputs = tf.image.resize_images(inputs, (256, 256))
    net = inputs
    net = tf.layers.conv2d(net, 32, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 64, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 128, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 256, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 512, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.layers.conv2d(net, 2014, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, training=training)
    net = contrib_layers.flatten(net)
    return tf.layers.dense(net, num_classes)


def mode_inception_v3(inputs, training, num_classes):
    inputs = tf.image.resize_images(inputs, (299, 299))
    logits, _ = slim_nets.resnet_v2.resnet_v2_50(inputs, num_classes=num_classes, is_training=training)
    logits = tf.squeeze(logits, (1, 2))
    return logits


def model_fn(features, labels, mode):
    if 'images/encoded' in features:
        inputs = tf.map_fn(preprocess_image, features['images/encoded'], dtype=tf.float32)
    else:
        inputs = features['images']
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
        inputs = (inputs - 0.5) * 2.0
    model = getattr(sys.modules[__name__], 'model_' + flags.model)

    logits = model(inputs, mode == tf.estimator.ModeKeys.TRAIN, len(CATEGORIES))

    predictions = tf.nn.softmax(logits)
    loss, train_op, metrics = None, None, None
    export_outputs = {
        'classified': tf.estimator.export.ClassificationOutput(
            scores=tf.identity(predictions, name="scores"),
            classes=tf.constant(CATEGORIES, dtype=tf.string, name='classes')
        )
    }

    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(labels, loss)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('image', inputs)
        for i, category in enumerate(CATEGORIES):
            tf.summary.image('image/' + category, tf.boolean_mask(inputs, tf.equal(labels, i)))

        batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(predictions, 1), labels), tf.float32), name='batch_accuracy')
        tf.summary.scalar('batch_accuracy', batch_accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = train.AdamOptimizer(learning_rate=flags.lr).minimize(loss, train.get_global_step())

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'accuracy': tf.metrics.accuracy(labels, tf.arg_max(predictions, 1))}

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=metrics,
                                      export_outputs=export_outputs)


def parse_input_images_to_infer():
    infer_image_path = flags.infer_image_path
    if not infer_image_path or not os.path.exists(infer_image_path):
        print('file not exist: {}'.format(infer_image_path))
        sys.exit(-1)

    if os.path.isdir(infer_image_path):
        image_files = [os.path.join(infer_image_path, f) for f in os.listdir(infer_image_path)]
    else:
        image_files = [infer_image_path]

    from scipy import misc
    images_data, parsed_image_files = [], []
    for image_file in image_files:
        try:
            image_data = misc.imread(image_file, mode='RGB')
            image_data = misc.imresize(image_data, (500, 500))
            images_data.append(image_data)
            parsed_image_files.append(image_file)
        except Exception:
            print('ignore unknown file: {}'.format(image_file))
    return np.array(images_data), parsed_image_files


def preprocess_image(image_buffer):
    image = tf.image.decode_png(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [500, 500], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def parse_input_data():
    def category_index(stat):
        if not stat['text']:
            return 0
        else:
            return 1 if len(CATEGORIES) == 3 else 0

    def parse_data(date_file):
        loaded = pickle.load(open(date_file, 'rb'))
        images, labels = loaded['images'], np.array(list(map(category_index, loaded['stats'])), dtype=np.int8)
        return images, labels

    if os.path.isdir(flags.data_path):
        date_files = [os.path.join(flags.data_path, f) for f in os.listdir(flags.data_path) if f.endswith('.pickle')]
        total_images, total_labels = [], []
        for data_file in date_files:
            _images, _labels = parse_data(data_file)
            total_images.append(_images)
            total_labels.append(_labels)
        images, labels = np.concatenate(tuple(total_images)), np.concatenate(tuple(total_labels))
    elif os.path.isfile(flags.data_path):
        images, labels = parse_data(flags.data_path)
    else:
        raise Exception('unknown file: {}'.format(flags.data_path))

    rand_idx = np.array(range(images.shape[0]))
    np.random.shuffle(rand_idx)
    images = images[rand_idx]
    labels = labels[rand_idx]
    return images, labels


def print_infer_result(batch_scores, image_files):
    image_idx = 0
    for scores in batch_scores:
        if max(scores) < 0.8:
            continue
        print('{}({}): {}'.format(
            image_files[image_idx], CATEGORIES[np.argmax(scores)],
            ', '.join(['{} - {:.3f}'.format(category, score) for category, score in zip(CATEGORIES, scores)])
        ))
        image_idx += 1


def main(_):
    global CATEGORIES
    CATEGORIES = flags.categories.split(',')
    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./data/models/{}'.format(flags.name),
        config=tf.estimator.RunConfig.replace(keep_checkpoint_max=20))

    if flags.mode in ['train', 'eval']:
        images, labels = parse_input_data()

        num_epochs = flags.num_epochs if flags.mode == 'train' else 1
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': images}, y=labels, batch_size=flags.batch_size, num_epochs=num_epochs, shuffle=True)

        if flags.mode == 'train':
            estimator.train(input_fn)
        elif flags.mode == 'eval':
            estimator.evaluate(input_fn)
    elif flags.mode == 'infer':
        images_data, image_files = parse_input_images_to_infer()
        input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': images_data}, batch_size=flags.batch_size, shuffle=False)

        batch_scores = estimator.predict(input_fn)
        print_infer_result(batch_scores, image_files)
    else:
        feature_spec = {'images/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string)}
        input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel('./data/saved_models/{}'.format(flags.name), input_fn)


if __name__ == '__main__':
    tf.app.run(main)
