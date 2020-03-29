"""
Utility functions used in other files.
"""

import tensorflow as tf


def _ones(n):
    return [1] * n


def repeat(x, c):
    return tf.reshape(
        tf.tile(tf.expand_dims(x, axis=1), [1, c, 1] + _ones(len(x.shape) - 2)),
        [x.shape[0], -1] + x.shape.as_list()[2:]
    )


def tile(x, c):
    return tf.reshape(
        tf.tile(tf.expand_dims(x, axis=2), [1, 1, c] + _ones(len(x.shape) - 2)),
        [x.shape[0], -1] + x.shape.as_list()[2:]
    )


def sample_k_distinct(num, weights, sample=False, sample_distinct=True):
    """
    Given a probability distribution, sample with replacement until there are
    `num` distinct elements.
    :param num: tensor ()
    :param weights: (?,)
    :param sample: If false, use top-K sampling.
    :param sample_distinct: If false, returns `num` elements which may contain
    repetitions. If true, only sample distinct elements.
    :return: indices, weights: (?,), (?,)
    """
    assert len(weights.shape) == 1
    seed = 89
    tf.set_random_seed(seed)

    if not sample:
        # Taking top-K from weights.
        weights, indices = tf.nn.top_k(weights, k=num)
        weights = weights - tf.reduce_logsumexp(weights, 0, keepdims=True)
        return indices, weights

    # Run probability sampling
    seed = tf.contrib.distributions.SeedStream(seed=seed, salt='one_hot_categorical')

    if not sample_distinct:
        indices = tf.random.categorical(tf.expand_dims(weights, axis=0), num)
        indices = tf.squeeze(indices, axis=0)
        weights = tf.fill((num,), -tf.log(tf.cast(num, tf.float32)))
        return indices, weights

    distr = tf.contrib.distributions.OneHotCategorical(logits=weights)
    vec0 = tf.zeros_like(weights, dtype=tf.int32)
    c = lambda v: tf.greater(num, tf.count_nonzero(v, dtype=tf.int32))
    b = lambda v: v + distr.sample(seed=seed())
    # Each value in vector counts how many times that index has been sampled
    vector = tf.while_loop(c, b, [vec0])

    mask = tf.greater(vector, 0)
    weights = tf.cast(tf.boolean_mask(vector, mask), tf.float32)  # Integers representing weights
    weights_sum = tf.reduce_sum(weights)
    weights = tf.log(weights) - tf.log(weights_sum)
    indices = tf.squeeze(tf.cast(tf.where(mask), tf.int32), axis=-1)

    indices = tf.reshape(indices, (num,))
    weights = tf.reshape(weights, (num,))
    return indices, weights
