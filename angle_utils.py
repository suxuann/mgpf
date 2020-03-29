"""
Utility functions for angles.
"""

import numpy as np


def normalize_rad(r):
    """
    Normalize a radian to between -np.pi and np.pi.
    """
    return np.mod(r + np.pi, 2 * np.pi) - np.pi


def rotation_rads(r):
    """
    Given an angle or an array of angles in radian, give the 4 perpendicular angles.
    :param r int or (K, 1)
    :return [int] * 4 or (K, 4)
    """
    try:
        float(r)
        output = [r, r + np.pi / 2., r + np.pi, r + np.pi * 3. / 2.]
        return np.sort([normalize_rad(r) for r in output])
    except TypeError:
        assert len(r.shape) == 2 and r.shape[1] == 1
        output = [r, r + np.pi / 2., r + np.pi, r + np.pi * 3. / 2.]
        output = np.concatenate(output, axis=-1)
        output = np.sort(normalize_rad(output), axis=1)
        return output
