import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
    inputs = tf.layers.batch_normalization(inputs=inputs,
                                           axis=1 if data_format == 'channels_first' else 3,
                                           momentum=_BATCH_NORM_DECAY,
                                           epsilon=_BATCH_NORM_EPSILON,
                                           center=True,
                                           scale=True,
                                           training=is_training,
                                           fused=True)
    return tf.nn.relu(inputs)


def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0],
                                        [0, 0],
                                        [pad_begin, pad_end],
                                        [pad_begin, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0],
                                        [pad_begin, pad_end],
                                        [pad_begin, pad_end],
                                        [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=('SAME' if strides == 1 else 'VALID'),
                            use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer(),
                            data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=3, strides=1, data_format=data_format)
    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=1, strides=1, data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(inputs, filters, kernel_size=3, strides=strides, data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(inputs, 4 * filters, kernel_size=1, strides=1, data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name, data_format):
    # Bottleneck blocks end with 4x the number of tilers as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs, filters_out, kernel_size=1, strides=strides, data_format=data_format)

    # Only the first block per block_layer used projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)
    return tf.identity(inputs, name)


def cifar10_resnet_v2_generator(resnet_size, num_classes, data_format=None):
    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    numblocks = (resnet_size - 2) // 6

    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(inputs=inputs, filters=16, kernel_size=3, strides=1, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = block_layer(inputs=inputs, filters=16, block_fn=building_block, blocks=numblocks,
                             strides=1, is_training=is_training, name='block_layer1', data_format=data_format)

        inputs = block_layer(inputs=inputs, filters=32, block_fn=building_block, blocks=numblocks,
                             strides=2, is_training=is_training, name='block_layer2', data_format=data_format)

        inputs = block_layer(inputs=inputs, filters=64, block_fn=building_block, blocks=numblocks,
                             strides=2, is_training=is_training, name='block_layer3', data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(inputs, pool_size=8, strides=1, padding='VALID', data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        inputs = tf.reshape(inputs, [-1, 64])
        inputs = tf.layers.dense(inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def imagenet_resnet_v2_generator(block_fn, layers, num_classes, data_format=None):
    if data_format is None:
        data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        inputs = block_layer(inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                             strides=1, is_training=is_training, name='block_layer1', data_format=data_format)

        inputs = block_layer(inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                             strides=2, is_training=is_training, name='block_layer2', data_format=data_format)

        inputs = block_layer(inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                             strides=2, is_training=is_training, name='block_layer3', data_format=data_format)

        inputs = block_layer(inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                             strides=2, is_training=is_training, name='block_layer4', data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(inputs, pool_size=7, strides=1, padding='VALID', data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')

        inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]},
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    params = model_params[resnet_size]
    return imagenet_resnet_v2_generator(params['block'], params['layers'], num_classes, data_format)
