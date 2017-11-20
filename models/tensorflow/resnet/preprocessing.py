import tensorflow as tf

_R_MEAN = 123.68 / 255
_G_MEAN = 116.78 / 255
_B_MEAN = 103.94 / 255

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    origin_shape = tf.shape(image)
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, origin_shape[2]])

    size_assertion = tf.Assert(tf.logical_and(
        tf.assert_greater(origin_shape[0], crop_height),
        tf.assert_greater(origin_shape[1], crop_width)
    ), ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _random_crop(images_list, crop_height, crop_width):
    if not images_list:
        raise ValueError('Empty image list.')

    rank_assertions = []
    for i in range(len(images_list)):
        image_rank = tf.rank(images_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                ['Wrong rank for tensor %s [expected] [actual]', images_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)
    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(images_list[0])

    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height),
                                                tf.greater_equal(image_width, crop_width)),
                                 ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(images_list)):
        image = images_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(tf.equal(height, image_height),
                                  ['Wrong height for tensor %s [expected] [actual]', image.name, height, image_height])
        width_assert = tf.Assert(tf.equal(width, image_width),
                                 ['Wrong width for tensor %s [expected][actual]', image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in images_list]


def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width, crop_height, crop_width))


def _mean_image_subtraction(image, means):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, tf.float32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.square(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side):
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN, resize_side_max=_RESIZE_SIDE_MAX):
    if is_training:
        return preprocess_for_train(image, output_height, output_width, resize_side_min, resize_side_max)
    else:
        return preprocess_for_eval(image, output_height, output_width, resize_side_min)
