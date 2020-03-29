"""
Utilities for processing maps.
"""

import cv2
import numpy as np
import tensorflow as tf

num_directions = 4


def decode_image(img_str, resize=None):
    """
    Decode image from tfrecord data
    :param img_str: image encoded as a png in a string
    :param resize: tuple width two elements that defines the new size of the image. optional
    :return: image as a numpy array
    """
    nparr = np.fromstring(img_str, np.uint8)
    img_str = cv2.imdecode(nparr, -1)
    if resize is not None:
        img_str = cv2.resize(img_str, resize)
    return img_str


def identify_margin(map_wall):
    """
    Each wall map in our dataset contains margin walls.
    This function identifies those walls and returns it as a separate map.
    :param map_wall: (H, W, C)
    :return map margins where walls are 1s.
    """
    rows = set()
    cols = set()
    # Here, 1s represent walls
    wall_val = 1
    height = map_wall.shape[0]

    # Start scanning from two different sides to remove margin walls
    max_diff = 1
    row_max, col_max = height, height
    for i in range(height):
        row_vals = map_wall[i, :]
        row_wall_ids = np.where(np.equal(row_vals, wall_val))[0]
        if len(row_wall_ids) > len(row_vals) - max_diff and i < row_max:
            rows.add(i)
        else:
            row_max = i
        col_vals = map_wall[:, i]
        col_wall_ids = np.where(np.equal(col_vals, wall_val))[0]
        if len(col_wall_ids) > len(col_vals) - max_diff and i < col_max:
            cols.add(i)
        else:
            col_max = i

    row_max, col_max = -1, -1
    for i in range(height):
        i = height - i - 1
        row_vals = map_wall[i, :]
        row_wall_ids = np.where(np.equal(row_vals, wall_val))[0]
        if len(row_wall_ids) > len(row_vals) - max_diff and i > row_max:
            rows.add(i)
        else:
            row_max = i
        col_vals = map_wall[:, i]
        col_wall_ids = np.where(np.equal(col_vals, wall_val))[0]
        if len(col_wall_ids) > len(col_vals) - max_diff and i > col_max:
            cols.add(i)
        else:
            col_max = i

    # Here, 1s represent walls.
    output = np.zeros_like(map_wall)
    output[list(rows), :] = 1.
    output[:, list(cols)] = 1.
    return output


def get_maps(global_maps, special_maps):
    """
    :return: wall_map, gaussian_map
    """
    empty_space_map = tf.expand_dims(tf.gather(special_maps, 0, axis=-1), axis=-1)  # 1 for empty spaces
    margin_map = tf.expand_dims(tf.gather(special_maps, 1, axis=-1), axis=-1)  # 1 for margins
    wall_map = tf.expand_dims(tf.gather(global_maps, 0, axis=-1), axis=-1)  # 1 for wall
    door_map = tf.expand_dims(tf.gather(global_maps, 1, axis=-1), axis=-1)

    gaussian_map = empty_space_map + wall_map - margin_map  # Possible empty space for Gaussian functions
    wall_map = wall_map - margin_map

    # """
    # Remove thick wall areas in wall and door maps.
    # Use a convolution kernel to identify all those areas.
    # """
    # ksize = 13
    # ksum = ksize * ksize
    # kernel = tf.ones((ksize, ksize, 1, 1), tf.float32)
    # wall_map_conv = tf.nn.depthwise_conv2d(
    #     wall_map, kernel, strides=[1, 1, 1, 1], padding='SAME'
    # )
    # door_map_conv = tf.nn.depthwise_conv2d(
    #     door_map, kernel, strides=[1, 1, 1, 1], padding='SAME'
    # )
    # zero_map = tf.zeros_like(wall_map)
    # wall_map = tf.where(tf.equal(wall_map_conv, ksum), zero_map, wall_map)
    # door_map = tf.where(tf.equal(door_map_conv, ksum), zero_map, door_map)
    # """End wall map."""

    door_map = tf.tile(tf.squeeze(door_map, axis=-1), (num_directions, 1, 1))
    wall_map = tf.tile(tf.squeeze(wall_map, axis=-1), (num_directions, 1, 1))
    gaussian_map = tf.tile(tf.squeeze(gaussian_map, axis=-1), (num_directions, 1, 1))
    return wall_map, door_map, gaussian_map


def convolve(wall_map, door_map, gaussian_map, points):
    """
    Apply filters generated from depth image to the global map. Return the resulting convolved maps.
    :param wall_map: (num_directions, H, W)
    :param door_map: (num_directions, H, W)
    :param gaussian_map: (num_directions, H, W)
    :param points: (num_directions, 56, 2)
    :return (num_directions, map_size, map_size, 1)
    """
    num_points = tf.shape(points)[1]
    map_size = tf.shape(wall_map)[1]
    wall_map = wall_map + door_map
    one_map, zero_map = tf.ones_like(wall_map), tf.zeros_like(wall_map)
    wall_map = tf.where(tf.greater(wall_map, 0), one_map, zero_map)

    def sum_conv(acc, ps):
        """
        Use translation and summation of global maps to achieve the same effect as convolution.
        Summation is done in while loop.
        :param acc: (num_directions, size, size, 1)
        :param curr: (num_directions, 2)
        :return:
        """
        eps = 1e-2
        outputs = list()
        ps = tf.reshape(ps, (-1, 2))  # (4, 2)
        used_map = wall_map[0, :, :]
        zero_map = tf.zeros_like(used_map)

        for i in range(num_directions):
            translation = -ps[i, :]
            conv_map = tf.cond(
                tf.less(tf.reduce_sum(tf.abs(translation)), eps),
                lambda: zero_map,
                lambda: tf.contrib.image.translate(used_map, translation),
            )
            conv_map = tf.expand_dims(conv_map, axis=0)
            outputs.append(conv_map)
        outputs = tf.expand_dims(tf.concat(outputs, axis=0), axis=-1)
        return acc + outputs

    # Loop on num_points to get the results of convolution
    i0 = tf.constant(0)
    conv_map0 = tf.zeros((num_directions, map_size, map_size, 1))
    c = lambda i, prev: i < num_points
    bw = lambda i, prev: (i + 1, sum_conv(prev, tf.gather(points, i, axis=1)))
    _, conv_maps = tf.while_loop(c, bw, (i0, conv_map0))

    conv_maps = tf.reshape(conv_maps, (num_directions, map_size, map_size))
    zero_maps = tf.zeros_like(conv_maps)
    conv_maps = tf.where(tf.greater(gaussian_map, 0), conv_maps, zero_maps)
    conv_maps = tf.where(tf.greater(wall_map + door_map, 0), zero_maps, conv_maps)
    return conv_maps


def is_filter_valid(filter_points):
    """
    Given a set of relative points representing the filter, returns a boolean value representing whether
    or not the filter / observation is valid. Invalid filters are those whose values are all 255s.
    :param filter_points: (num_directions, 56, 2)
    :return: tf.bool indicating whether the filter points are valid.
    """
    epsilon = 1e-2
    num_points = tf.cast(tf.shape(filter_points)[1], tf.float32)
    distances = tf.reduce_sum(tf.abs(filter_points[0]), axis=1)  # (56,)
    invalids = tf.reduce_sum(tf.cast(tf.less(distances, epsilon), tf.float32))
    # The filter is valid if the number of invalid points is below a certain fraction
    fraction = 1. / 5.
    is_valid = tf.less_equal(invalids, tf.cast(tf.multiply(num_points, fraction), tf.float32))
    return is_valid
