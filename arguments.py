"""
Settings for running MGPF. Settings used in the paper are defined in /configs/ folder.
"""
import configargparse
import numpy as np


def parse_args(args=None):
    """
    Parse command line arguments
    :param args: command line arguments or None (default)
    :return: dictionary of parameters
    """
    p = configargparse.ArgParser(default_config_files=[])

    p.add('-c', '--config', required=True, is_config_file=True, help='Config file, located under ./config/')
    p.add('--testfiles', nargs='*', help='Data file(s) for validation or evaluation (tfrecord).')

    # Input configuration
    p.add('--obsmode', type=str, default='depth', help='Observation input type. Possible values: depth.')
    p.add('--mapmode', type=str, default='wall',
          help='Map input type with different (semantic) channels. ' +
               'Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype')
    p.add('--map_pixel_in_meters', type=float, default=0.02,
          help='The width (and height) of a pixel of the map in meters. Defaults to 0.02 for House3D data.')

    p.add('--init_particles_distr', type=str, default='tracking',
          help='Distribution of initial particles. Possible values: tracking / one-room / two-rooms / all-rooms')
    p.add('--init_particles_std', nargs='*', default=["0.3", "0.523599"],  # tracking setting, 30cm, 30deg
          help='Standard deviations for generated initial particles. Only applies to the tracking setting.' +
               'Expects two float values: translation std (meters), rotation std (radians)')
    p.add('--trajlen', type=int, default=24,
          help='Length of trajectories. Assumes lower or equal to the trajectory length in the input data.')

    # MGPF configuration
    p.add('--num_particles', type=int, default=30, help='Number of particles in PF-net.')
    p.add('--transition_std', nargs='*', default=["0.0", "0.0"],
          help='Standard deviations for transition model. Expects two float values: ' +
               'translation std (meters), rotatation std (radians). Defaults to zeros.')
    p.add('--model', type=str, default='GPF', help='Which model to use: GPF / PF.')

    # In the code I'm taking 2 times these values, then take square as the covariance.
    p.add('--std_pos_max', type=float, default=500., help='Deviation in the x, y direction for particles')
    p.add('--std_theta_max', type=float, default=np.pi / 2., help='Deviation in the theta direction for particles')

    p.add('--std_pos_obs', type=float, default=5., help='Deviation in the x, y direction for observations')
    p.add('--std_theta_obs', type=float, default=np.pi / 3., help='Deviation in the theta direction')

    # Training configuration
    p.add('--batchsize', type=int, default=1, help='Minibatch size for training. Must be 1 for evaluation.')
    p.add('--gpu', type=int, default=0, help='Which GPU to use to run evaluation.')
    p.add('--seed', type=int, default=23,
          help='Fix the random seed of numpy and tensorflow if set to larger than zero.')

    params = p.parse_args(args=args)

    # Fix numpy seed if needed
    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)

    # Convert multi-input fileds to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    return params
