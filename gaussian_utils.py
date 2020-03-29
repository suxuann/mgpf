"""
Utility functions for generating Gaussians.
"""
import numpy as np
import tensorflow as tf


def normalize_weight(x):
    """
    Normalize 1D probability or logits array to 1D logits array.
    """
    assert len(x.shape.as_list()) == 1
    s = tf.reduce_sum(x)
    output = tf.log(x) - tf.log(s)
    return output


class GU:
    def __init__(self, params):
        self.params = params

    def cov(self, points_x, points_y, weights, center):
        """
        Compute covariance of the matrix formed by all given points. Weights are not used, to save computation.
        :param points_x, points_y, weights: (?,)
        """
        assert points_x.shape.as_list() == points_y.shape.as_list() == weights.shape.as_list()
        assert len(center.shape.as_list()) == 1

        # Take square of max distance to the center point as the covariance.
        std_x = (tf.reduce_max(tf.abs(points_x - center[0])) + self.params.std_pos_obs)
        std_y = (tf.reduce_max(tf.abs(points_y - center[1])) + self.params.std_pos_obs)
        zero = tf.zeros_like(std_x)
        output = tf.reshape(tf.stack([std_x * std_x, zero, zero, std_y * std_y]), (2, 2))
        return output

    def gaussian_from_points(self, image, conv_max, points, ratio=5):
        """
        Image: a probability distribution over 2D space. Generates a Gaussian function to approximate the distribution.
        Mean is taken as local maximums on the image.
        Covariance matrix is computed according to the standard formula with weights for each point.
        :param image: (H, W)
        :param conv_max: downsampled image with a ratio, to determine local maximums
        :param points: (?, 2)
        :return center (2,); covariance (2, 2); weight ()
        """
        assert len(image.shape.as_list()) == 2
        assert len(points.shape.as_list()) == 2

        point_weights = tf.cast(tf.gather_nd(image, points), tf.float32)  # (?)
        points = tf.cast(points, tf.float32)

        points_x = tf.gather(points, 0, axis=-1)
        points_y = tf.gather(points, 1, axis=-1)

        """Gaussian centers: take a local maximum as center."""
        points = tf.cast(tf.round(tf.divide(points, ratio)), tf.int32)
        points = tf.minimum(points, tf.shape(conv_max)[0] - 1)
        local_maxs = tf.gather_nd(conv_max, points)
        local_maxs = tf.squeeze(tf.where(tf.greater(local_maxs, 0)), axis=-1)
        points = tf.cond(
            tf.greater(tf.shape(local_maxs)[0], 0),
            lambda: tf.gather(points, local_maxs, axis=0),
            lambda: points
        )
        center = tf.gather(points, tf.floordiv(tf.shape(points)[0], 2)) * ratio
        center = tf.cast(center, tf.float32)
        """End centers."""

        covariance = self.cov(points_x, points_y, point_weights, center)
        weight = tf.fill((1,), tf.reduce_max(point_weights))

        center = tf.reverse(center, axis=[0])
        covariance = tf.reverse(covariance, axis=[0, 1])
        return center, covariance, weight

    def threshold(self, image):
        """
        Perform image thresholding.
        :param image (H, W)
        """
        assert len(image.shape.as_list()) == 2
        frac = .6
        threshold = tf.reduce_max(image) * frac
        ones = tf.ones_like(image)
        zeros = tf.zeros_like(image)
        output = tf.where(tf.greater(image, threshold), ones, zeros)
        return output

    def segment(self, image):
        """
        Identify connected components (typically rectangles) in the given image
        after thresholding and Gaussian smoothing.
        :param image (H, W)
        """
        assert len(image.shape.as_list()) == 2
        thresholded = self.threshold(image)
        labeled = tf.contrib.image.connected_components(thresholded)
        ncomponent = tf.reduce_max(labeled)
        return labeled, ncomponent

    def gaussians_from_labeled_components(self, conv_map, conv_max, angle, labeled, ncomponent):
        """
        Return Gaussians from components in the labeled image.
        :param conv_map, labeled: (H, W)
        :param angle: (1)
        :return:
        """
        assert len(conv_map.shape.as_list()) == len(labeled.shape.as_list()) == 2
        assert len(angle.shape.as_list()) == 0
        # Extend center and covariance from 2x2 to 3x3
        extend_center = lambda c: tf.expand_dims(tf.concat([c, [angle]], axis=-1), axis=0)

        def extend_covariance(cov):
            # Extend covariance matrix from 2x2 to 3x3.
            output = tf.concat([cov, tf.zeros((1, 2))], axis=0)
            row = tf.constant([[0.], [0.], [np.square(self.params.std_theta_obs)]])
            output = tf.concat([output, row], axis=-1)
            output = tf.expand_dims(output, axis=0)
            return output

        def body(j, centers, covariances, weights):
            """
            Generate Gaussians in j-th connected component.
            """
            j = j + 1
            points = tf.where(tf.equal(labeled, j))
            points_x = tf.gather(points, 0, axis=-1)
            points_y = tf.gather(points, 1, axis=-1)

            # Filter out Gaussians that are too small
            cutoff = 5
            num_points = 4
            points_diff = [tf.reduce_max(points_x) - tf.reduce_min(points_x),
                           tf.reduce_max(points_y) - tf.reduce_min(points_y)]
            filtered = tf.logical_and(tf.less_equal(points_diff[0], cutoff),
                                      tf.less_equal(points_diff[1], cutoff))
            filtered = tf.logical_or(filtered, tf.less_equal(tf.shape(points)[0], num_points))

            def true_fn():
                return j, centers, covariances, weights

            def _unfiltered(j, centers, covariances, weights):
                # One Gaussian from one set of points
                _center, _covariance, _weight = self.gaussian_from_points(conv_map, conv_max, points)

                centers = tf.concat([centers, extend_center(_center)], axis=0)
                covariances = tf.concat([covariances, extend_covariance(_covariance)], axis=0)
                weights = tf.concat([weights, _weight], axis=0)
                return j, centers, covariances, weights

            return tf.cond(
                filtered,
                true_fn,
                lambda: _unfiltered(j, centers, covariances, weights)
            )

        def init_gaussians(ps):
            """
            Given points from the first component, generate the first Gaussian function.
            """
            _cent, _cov, _w = self.gaussian_from_points(conv_map, conv_max, ps)
            _cent = extend_center(_cent)
            _cov = extend_covariance(_cov)
            return _cent, _cov, _w

        def placeholder_fn():
            """
            Return placeholder values when number of components is 0.
            """
            centers = tf.zeros((0, 3))
            covariances = tf.zeros((0, 3, 3))
            weights = tf.zeros((0,))
            return centers, covariances, weights

        def true_fn():
            """Generate proper Gaussians."""
            j0 = tf.constant(1)
            points0 = tf.where(tf.equal(labeled, j0))
            centers0, covariances0, weights0 = init_gaussians(points0)

            c = lambda j, _, __, ___: tf.greater(ncomponent, j)
            j, centers, covariances, weights = tf.while_loop(
                c, body, [j0, centers0, covariances0, weights0],
                shape_invariants=[j0.get_shape(), tf.TensorShape([None, 3]),
                                  tf.TensorShape([None, 3, 3]), tf.TensorShape([None, ])]
            )
            return centers, covariances, weights

        return tf.cond(tf.equal(ncomponent, 0), placeholder_fn, true_fn)

    def local_maximum(self, conv_maps, ratio=5):
        """
        Get all the local maximums in the given convolution maps.
        :param conv_maps: (num_directions, size, size,)
        :param ratio: downsampling ratio
        :return: conv_maxs (num_directions, size, size)
        """
        conv_maps = tf.expand_dims(conv_maps, axis=-1)
        conv_maps = tf.nn.pool(conv_maps, window_shape=(ratio, ratio),
                               strides=(ratio, ratio), pooling_type='MAX', padding='SAME')
        pool_maxs = tf.nn.pool(conv_maps, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        conv_maxs = tf.where(tf.equal(conv_maps, pool_maxs), conv_maps, tf.zeros_like(conv_maps))

        threshold = tf.reduce_max(conv_maxs) * 0.8
        zero_maps = tf.zeros_like(conv_maxs)
        conv_maxs = tf.where(tf.greater_equal(conv_maxs, threshold), conv_maxs, zero_maps)
        conv_maxs = tf.squeeze(conv_maxs, axis=-1)
        return conv_maxs

    def gaussians_from_conv_map(self, conv_maps, angles):
        """
        Generate a set of Gaussians from the given convolution maps for 4 angles.
        :param conv_maps: (num_directions, size, size,)
        :param local_maxs: (?, 3) local maximums as Gaussian centers
        :param angles: (num_directions)
        """
        assert len(conv_maps.shape.as_list()) == 3
        assert len(angles.shape.as_list()) == 1

        states = list()
        covariances = list()
        weights = list()
        conv_maxs = self.local_maximum(conv_maps)

        labeled_list = list()
        num_directions = conv_maps.shape.as_list()[0]
        for i in range(num_directions):
            angle = tf.gather(angles, i)
            conv_map = tf.gather(conv_maps, i)
            conv_max = tf.gather(conv_maxs, i)

            labeled, ncomponent = self.segment(conv_map)
            cent, cov, w = self.gaussians_from_labeled_components(
                conv_map, conv_max, angle, labeled, ncomponent
            )

            states.append(cent)
            covariances.append(cov)
            weights.append(w)
            labeled_list.append(tf.expand_dims(labeled, axis=0))

        states = tf.concat(states, axis=0)
        covariances = tf.concat(covariances, axis=0)
        weights = tf.concat(weights, axis=0)

        labeled_list = tf.concat(labeled_list, axis=0)
        weights = normalize_weight(weights)

        num_gaussians = tf.shape(states)[0]
        states = tf.reshape(states, (num_gaussians, 3))
        covariances = tf.reshape(covariances, (num_gaussians, 3, 3))
        weights = tf.reshape(weights, (num_gaussians,))
        return states, covariances, weights, labeled_list
