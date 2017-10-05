# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import scipy
import numpy as np
import os

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'image_to_infer', None, 'The image file path.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'infer_image_size', None, 'Infer image size')

tf.app.flags.DEFINE_string(
    'labels_file', './data/labels.txt', 'Label file path')

tf.app.flags.DEFINE_integer('batch_size', -1, 'Batch size.')

tf.app.flags.DEFINE_string('output_file', None, 'output file path.')

FLAGS = tf.app.flags.FLAGS


def read_images(image_files, image_size):
  images = []
  for image_file in image_files:
    image = scipy.misc.imread(image_file, mode='RGB')
    image = scipy.misc.imresize(image, (image_size, image_size))
    image = (image / 255 - 0.5) * 2
    images.append(image)
  return np.array(images, np.float32)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.datasets_map[FLAGS.dataset_name]
    labels = open(FLAGS.labels_file, 'r').read()
    labels = [t.strip() for t in labels.split('\n') if t.strip()]
    labels = dict([l.split(':') for l in labels])

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.NUM_CLASSES - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    if os.path.isdir(FLAGS.image_to_infer):
      image_files = []
      for f in os.listdir(FLAGS.image_to_infer):
        image_files.append(os.path.join(FLAGS.image_to_infer, f))
    else:
        image_files = FLAGS.image_to_infer.split(',')
    infer_image_size = FLAGS.infer_image_size or network_fn.default_image_size

    ####################
    # Define the model #
    ####################
    batch_size = len(image_files) if FLAGS.batch_size == -1 else FLAGS.batch_size
    images_input = tf.placeholder(tf.float32, (None, infer_image_size, infer_image_size, 3))
    logits, _ = network_fn(images_input)
    predictions_k_value, predictions_k = tf.nn.top_k(logits, 3)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Infering %s' % checkpoint_path)

    saver = tf.train.Saver(slim.get_variables_to_restore())
    predictions_k_output = []
    predictions_k_detail_output = []

    with tf.Session() as session:
      saver.restore(session, checkpoint_path)
      num_epochs = math.ceil(len(image_files) / batch_size)
      for i in range(num_epochs):
        predictions_k_, predictions_value_k_ = session.run([predictions_k, predictions_k_value], {
          images_input: read_images(image_files[batch_size*i:batch_size*(i+1)], infer_image_size)
        })
        for j, pk in enumerate(predictions_k_):
          pkv = predictions_value_k_[j]
          pk_str = ', '.join(['%s(%s)' % (labels[str(p)], p) for p in pk])
          print('%s: %s' % (image_files[i*batch_size+j], pk_str))
          predictions_k_detail_output.append({
              'image_id': os.path.basename(image_files[i * batch_size + j], pk_str),
              "label_id":
          })
          # predictions_k_output.append({'image_id': os.path.basename(image_files[i*batch_size+j]), 'label_id': [int(pk[0]) for p in pk]})
          predictions_k_output.append({'image_id': os.path.basename(image_files[i * batch_size + j]), 'label_id': int(pk[0])})
    if FLAGS.output_file:
      import json
      open(FLAGS.output_file, 'w').write(json.dumps(predictions_k_output))
      tf.logging.info('write to file: %s', FLAGS.output_file)

if __name__ == '__main__':
  tf.app.run()
