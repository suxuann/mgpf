import os

import tensorflow as tf
from scipy.ndimage.filters import maximum_filter
from sklearn import linear_model
from tensorpack import dataflow
from tensorpack.dataflow.base import RNGDataFlow

import map_utils as mu
from angle_utils import *
from map_utils import decode_image

try:
    import ipdb as pdb
except Exception:
    import pdb

num_directions = 4


def raw_images_to_array(images):
    """
    Decode and normalize multiple images from tfrecord data
    :param images: list of images encoded as a png in a string
    :return: a numpy array of size (N, 56, 56, channels), normalized for training
    """
    image_list = []
    for image_str in images:
        image = decode_image(image_str, (56, 56))
        image = scale_observation(np.atleast_3d(image.astype(np.float32)))
        image = remove_depth_noise(image)
        image_list.append(image)

    return np.stack(image_list, axis=0)


def scale_observation(x):
    """
    Normalizes observation input, either an rgb image or a depth image
    :param x: observation input as numpy array, either an rgb image or a depth image
    :return: numpy array, a normalized observation
    """
    if x.ndim == 2 or x.shape[2] == 1:  # Depth
        return x  # Depth is not scaled.
    else:  # RGB
        return x * (2.0 / 255.0) - 1.0


def bounding_box(img):
    """
    Bounding box of non-zeros in an array (inclusive). Used with 2D maps
    :param img: numpy array
    :return: inclusive bounding box indices: top_row, bottom_row, leftmost_column, rightmost_column
    """
    # helper function to
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


class BatchDataWithPad(dataflow.BatchData):
    """
    Stacks datapoints into batches. Selected elements can be padded to the same size in each batch.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False, padded_indices=()):
        """
        :param ds: input dataflow. Same as BatchData
        :param batch_size: mini batch size. Same as BatchData
        :param remainder: if data is not enough to form a full batch, it makes a smaller batch when true.
        Same as BatchData.
        :param use_list: if True, components will contain a list of datapoints instead of creating a new numpy array.
        Same as BatchData.
        :param padded_indices: list of filed indices for which all elements will be padded with zeros to mach
        the largest in the batch. Each batch may produce a different size datapoint.
        """
        super(BatchDataWithPad, self).__init__(ds, batch_size, remainder, use_list)
        self.padded_indices = padded_indices

    def get_data(self):
        """
        Yields:  Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False, padded_indices=()):
        """
        Re-implement the parent function with the option to pad selected fields to the largest in the batch.
        """
        assert not use_list  # cannot match shape if they must be treated as lists
        size = len(data_holder[0])
        result = []
        for k in range(size):
            dt = data_holder[0][k]
            if type(dt) in [int, bool]:
                tp = 'int32'
            elif type(dt) == float:
                tp = 'float32'
            else:
                try:
                    tp = dt.dtype
                except AttributeError:
                    raise TypeError("Unsupported type to batch: {}".format(type(dt)))
            try:
                if k in padded_indices:
                    # pad this field
                    shapes = np.array([x[k].shape for x in data_holder], 'i')  # assumes ndim are the same for all
                    assert shapes.shape[1] == 3  # only supports 3D arrays for now, e.g. images (height, width, ch)
                    matching_shape = shapes.max(axis=0).tolist()
                    new_data = np.zeros([shapes.shape[0]] + matching_shape, dtype=tp)
                    for i in range(len(data_holder)):
                        shape = data_holder[i][k].shape
                        new_data[i, :shape[0], :shape[1], :shape[2]] = data_holder[i][k]
                    result.append(new_data)
                else:
                    # no need to pad this field, simply create batch
                    result.append(np.asarray([x[k] for x in data_holder], dtype=tp))
            except Exception as e:
                # exception handling. same as in parent class
                print(e.message)
                pdb.set_trace()
                dataflow.logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                if isinstance(dt, np.ndarray):
                    s = dataflow.pprint.pformat([x[k].shape for x in data_holder])
                    dataflow.logger.error("Shape of all arrays to be batched: " + s)
                try:
                    # open an ipython shell if possible
                    import IPython as IP;
                    IP.embed()  # noqa
                except ImportError:
                    pass
        return result


def camera_angles(width):
    """
    Generate `width' (typically 56) angles for each column of a camera image.
    """
    focal_len = width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
    dist = np.arange(width) + 0.5 - width / 2.0
    angles = np.arctan2(dist, focal_len)
    return angles


def distances_from_depth_image(depth):
    """
    Given a depth image, perform a simulated laser scan to get the distances.
    There's some heuristic processing.
    :param depth: (height, width)
    :return: distances of each column of depth image
    """
    invalid_val = 255
    depth_to_grid = 3.9
    height, width = depth.shape

    focal_len = width / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
    dist = np.arange(width) + 0.5 - width / 2.0

    percentile = 80
    dists = np.array([])
    for i in range(height):
        col = np.copy(depth[:, i])

        # Invalid indices
        inv_indices = np.where(col == invalid_val)[0]
        if len(inv_indices) > 0:
            inv_last = inv_indices[-1]
            inv_bool = inv_last == col.shape[0] - 1  # Bottom of image is 255

            if not inv_bool:
                ys = col[inv_last + 1:inv_last + 9]
                diffs = np.abs(np.diff(ys))
                inv_bool = np.sum(np.greater(diffs, 4).astype(np.int32))
                inv_bool = ys[0] >= 100 and inv_bool >= 2  # Roughly detect a window

            if inv_bool:
                d = 0
                dists = np.append(dists, d)
                continue

        # Process remaining values
        indices = np.where((col < invalid_val) & (col > 0))[0]
        d = 0
        if len(indices):
            d = depth_to_grid * np.percentile(col[indices], percentile)
        dists = np.append(dists, d)

    dists_ths = np.max(dists) * 0.2
    dists[dists < dists_ths] = 0.
    dists = np.sqrt(np.square(dists * dist / focal_len) + np.square(dists))
    return dists


def filter_points_from_depth_image(depth):
    """
    Given a depth image, there are 4 angles to rotate its filter so that the resulting edges align
    with the map. Compute the angles and relative coordinates of the points.
    :param depth: (H, W) image
    :return angles (4,)
    points on wall (4, 56, 2)
    """
    # Compute a RANSAC regression for points on the depth image to give the direction of one edge
    height, width = depth.shape
    dists = distances_from_depth_image(depth)

    assert len(dists) == width
    _camera_angles = camera_angles(width)
    px = np.multiply(dists, np.sin(_camera_angles))
    py = np.multiply(dists, np.cos(_camera_angles))

    ransac = linear_model.RANSACRegressor(residual_threshold=5.)
    eps = .1
    px = px[dists > eps]  # Need to filter out the zero values
    py = py[dists > eps]
    if len(px) > 1:
        ransac.fit(px.reshape(-1, 1), py.reshape(-1, 1))
        y = ransac.predict([[1], [2]])
        angle = normalize_rad(np.arctan(y[1][0] - y[0][0]))
    else:
        angle = 0

    # Produce outputs
    rotated_angles = rotation_rads(angle)
    rotated_angles = np.asarray(np.sort(rotated_angles))

    points_wall = list()
    for angle in rotated_angles:
        angles = [normalize_rad(a + angle + np.pi / 2.) for a in _camera_angles]
        angles_sin = np.sin(angles)
        angles_cos = np.cos(angles)

        px = np.expand_dims(np.multiply(dists, angles_sin), axis=-1)
        py = -np.expand_dims(np.multiply(dists, angles_cos), axis=-1)
        ps = np.concatenate([px, py], axis=-1)
        points_wall.append(ps)

    points_wall = np.asarray(points_wall)
    return rotated_angles, points_wall


def remove_depth_noise(depth):
    """
    Remove noise in depth image using a simple maximum filter.
    :param depth: (K, size, size, 1)
    """
    output = maximum_filter(depth, footprint=np.ones((3, 3, 1)))
    return output


class House3DFilterData(RNGDataFlow):
    """
    Process tfrecords data of House3D trajectories. Produces a dataflow with the following fields:
    true state, global map, observations
    """

    def __init__(self, files, mapmode, obsmode, trajlen, num_particles, init_particles_distr,
                 init_particles_cov, seed=None):
        """
        :param files: list of data file names. assumed to be tfrecords files
        :param mapmode: string, map type. Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype
        :param obsmode: string, observation type. Possible values: rgb / depth / rgb-depth. Vrf is not yet supported
        :param trajlen: int, length of trajectories
        :param seed: int or None. Random seed will be fixed if not None.
        """
        self.files = files
        self.mapmode = mapmode
        self.obsmode = obsmode
        self.trajlen = trajlen
        self.seed = seed
        self.num_particles = num_particles
        self.init_particles_distr = init_particles_distr
        self.init_particles_cov = init_particles_cov

        # Count total number of trajectories
        count = 0
        for f in self.files:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
            record_iterator = tf.python_io.tf_record_iterator(f)
            for _ in record_iterator:
                count += 1
        self.count = count

    def size(self):
        return self.count

    def reset_state(self):
        """ Reset state. Fix numpy random seed if needed."""
        super(House3DFilterData, self).reset_state()
        np.random.seed(self.seed)

    def get_data(self):
        """
        Yields datapoints, all numpy arrays, with the following fields.
        :return true states: (trajlen, 3). Second dimension corresponds to x, y, theta coordinates.
        :return global map: (n, m, ch). shape is different for each map. number of channels depend on the mapmode setting
        :return observations: (trajlen, 56, 56, ch) number of channels depend on the obsmode setting
        """
        for file in self.files:
            gen = tf.python_io.tf_record_iterator(file)
            for data_i, string_record in enumerate(gen):
                result = tf.train.Example.FromString(string_record)
                features = result.features.feature

                # Process maps
                map_wall = self.process_wall_map(features['map_wall'].bytes_list.value[0])
                map_door = self.process_door_map(features['map_door'].bytes_list.value[0])
                map_empty = self.process_empty_map(features['map_roomtype'].bytes_list.value[0])

                # Has been rescaled to 0..1 range. After transform, 1s represent walls.
                # Zero padding will produce the equivalent of empty space
                # map_rooms = self.process_roomtype_map(features['map_roomtype'].bytes_list.value[0])
                global_map_list = [map_wall, map_door]
                global_map = np.concatenate(global_map_list, axis=-1)

                if self.init_particles_distr == 'tracking':
                    map_roomid = None
                elif self.init_particles_distr == 'all-room':
                    map_roomid = map_empty
                else:
                    map_roomid = self.process_roomid_map(features['map_roomid'].bytes_list.value[0])

                map_margin = mu.identify_margin(global_map[:, :, :1])
                map_roomtype = self.process_roomtype_map(features['map_roomtype'].bytes_list.value[0])
                special_map_list = [map_empty, map_margin, map_roomtype]
                special_map = np.concatenate(special_map_list, axis=-1).astype(np.float32)

                # Process true states
                true_states = features['states'].bytes_list.value[0]
                true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))
                true_states = np.concatenate([true_states[:, :2], normalize_rad(true_states[:, 2:])], axis=-1)

                # Process odometry
                odometry = features['odometry'].bytes_list.value[0]
                odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

                # Trajectory may be longer than what we use for training
                data_trajlen = true_states.shape[0]
                assert data_trajlen >= self.trajlen
                true_states = true_states[:self.trajlen]
                odometry = odometry[:self.trajlen]

                # Process observations
                assert self.obsmode == 'depth'
                observation = raw_images_to_array(list(features['depth'].bytes_list.value)[:self.trajlen])

                # generate particle states
                init_particles = self.random_particles(
                    true_states[0], self.init_particles_distr,
                    self.init_particles_cov, self.num_particles,
                    roomidmap=map_roomid, seed=23
                )

                filter_angles = list()
                points = list()
                for i in range(self.trajlen):
                    depth_i = np.squeeze(observation[i], axis=-1)
                    angles_i, points_i = filter_points_from_depth_image(depth_i)
                    filter_angles.append(angles_i)
                    points.append(points_i)

                filter_angles = np.asarray(filter_angles)
                points = np.asarray(points)

                yield (true_states, init_particles, global_map, special_map, observation, odometry,
                       filter_angles, points, (True))

    def process_wall_map(self, map_feature):
        wall_map = np.atleast_3d(decode_image(map_feature))
        wall_map = np.transpose(wall_map, axes=[1, 0, 2])
        wall_map = wall_map.astype(np.float32) * (1.0 / 255.0)
        return wall_map

    def process_door_map(self, map_feature):
        wall_map = np.atleast_3d(decode_image(map_feature))
        wall_map = np.transpose(wall_map, axes=[1, 0, 2])
        wall_map = wall_map.astype(np.float32) * (1.0 / 255.0)
        return wall_map

    def process_empty_map(self, map_feature):
        # In the returned empty map, 1 represents empty space
        empty_map = np.atleast_3d(decode_image(map_feature))
        ones, zeros = np.ones_like(empty_map), np.zeros_like(empty_map)
        empty_map = np.where(np.greater(empty_map, 1), ones, zeros)
        empty_map = np.transpose(empty_map, axes=[1, 0, 2])
        return empty_map

    def process_roomid_map(self, roomidmap_feature):
        # this is not transposed, unlike other maps
        roomidmap = np.atleast_3d(decode_image(roomidmap_feature))
        return roomidmap

    def process_roomtype_map(self, map_feature):
        output = np.atleast_3d(decode_image(map_feature))
        # transpose and invert
        output = np.transpose(output, axes=[1, 0, 2])
        return output

    @staticmethod
    def random_particles(state, distr, particles_cov, num_particles, roomidmap, seed=None):
        """
        Generate a random set of particles
        :param state: true state, numpy array of x,y,theta coordinates
        :param distr: string, type of distribution. Possible values: tracking / one-room.
        For 'identity', simply return k true states as particles.
        For 'tracking' the distribution is a Gaussian centered near the true state.
        For 'one-room' the distribution is uniform over states in the room defined by the true state.
        For 'two-room' the distribution is over states in the same room as the true state as well as another room.
        For 'all-room' the distribution is over all rooms in the given map.
        :param particles_cov: numpy array of shape (3,3), defines the covariance matrix if distr == 'tracking'
        :param num_particles: number of particles
        :param roomidmap: numpy array, map of room ids. Values define a unique room id for each pixel of the map.
        :param seed: int or None. If not None, the random seed will be fixed for generating the particle.
        The random state is restored to its original value.
        :return: numpy array of particles (num_particles, 3)
        """
        assert distr in ['identity', 'tracking', 'localization', 'global',
                         'one-room', 'two-room', 'all-room']

        particles = np.zeros((num_particles, 3), np.float32)
        if distr == 'identity':
            state = np.expand_dims(state, axis=0)
            particles = np.tile(state, [num_particles, 1])
        elif distr == 'tracking':
            if seed is not None:
                random_state = np.random.get_state()
                np.random.seed(seed)

            # Sample offset from the Gaussian
            center = np.random.multivariate_normal(mean=state, cov=particles_cov)

            # Restore random seed
            if seed is not None:
                np.random.set_state(random_state)

            # Sample particles from the Gaussian, centered around the offset
            particles = np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles)
        elif distr == 'one-room' or distr == 'localization':
            # Mask the room the initial state is in
            masked_map = (roomidmap == roomidmap[int(np.rint(state[0])), int(np.rint(state[1]))])

            # Get bounding box for more efficient sampling
            rmin, rmax, cmin, cmax = bounding_box(masked_map)

            # Rejection sampling inside bounding box
            sample_i = 0
            while sample_i < num_particles:
                particle = np.random.uniform(low=(rmin, cmin, 0.0), high=(rmax, cmax, 2.0 * np.pi), size=(3,), )
                # Reject if mask is zero
                if not masked_map[int(np.rint(particle[0])), int(np.rint(particle[1]))]:
                    continue
                particles[sample_i] = particle
                sample_i += 1
        elif distr == 'two-room':
            rooms = np.unique(roomidmap)
            room_1 = roomidmap[int(np.rint(state[0])), int(np.rint(state[1]))]
            rooms_invalid = np.logical_or(rooms == 0, rooms == room_1)
            rooms_remain = np.delete(rooms, np.where(rooms_invalid), axis=0)
            room_2 = np.random.choice(rooms_remain) if len(rooms_remain) else room_1
            masked_map = np.logical_or(roomidmap == room_1, roomidmap == room_2)

            rmin, rmax, cmin, cmax = bounding_box(masked_map)

            sample_i = 0
            while sample_i < num_particles:
                particle = np.random.uniform(low=(rmin, cmin, 0.0), high=(rmax, cmax, 2.0 * np.pi), size=(3,), )
                if not masked_map[int(np.rint(particle[0])), int(np.rint(particle[1]))]:
                    continue
                particles[sample_i] = particle
                sample_i += 1
        elif distr == 'all-room' or distr == 'global':
            ys, xs = np.where(roomidmap[:, :, 0] > 0)
            indices = np.arange(len(xs))
            np.random.shuffle(indices)
            indices = indices[:num_particles]

            xs = np.expand_dims(np.take(xs, indices).astype(np.float32), axis=-1)
            ys = np.expand_dims(np.take(ys, indices).astype(np.float32), axis=-1)
            angles = np.random.uniform(low=0, high=2 * np.pi, size=(num_particles, 1))
            particles = np.concatenate([xs, ys, angles], axis=-1)
        else:
            raise ValueError

        return particles


def get_dataflow(files, params):
    """
    Build a tensorflow Dataset from appropriate tfrecords files.
    :param files: list a file paths corresponding to appropriate tfrecords data
    :param params: parsed arguments
    :param is_training: bool, true for training.
    :return: (nextdata, num_samples).
    nextdata: list of tensorflow ops that produce the next input with the following elements:
    true_states, global_map, init_particles, observations, odometries, is_first_step.
    See House3DTrajData.get_data for definitions.
    """
    mapmode = params.mapmode
    obsmode = params.obsmode
    batchsize = params.batchsize
    trajlen = params.trajlen
    num_particles = params.num_particles

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std[0] = particle_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    particle_std2 = np.square(particle_std)  # variance
    init_particles_cov = np.diag(particle_std2[(0, 0, 1),])

    df = House3DFilterData(
        files, mapmode, obsmode, trajlen, num_particles,
        params.init_particles_distr, init_particles_cov,
        seed=params.seed
    )

    df = dataflow.FixedSizeData(df, size=df.size(), keep_state=False)

    df = BatchDataWithPad(df, batchsize, padded_indices=(2, 3))

    num_samples = df.size()

    df.reset_state()

    # # test dataflow
    # df = dataflow.TestDataSpeed(dataflow.PrintData(df), 100)
    # df.start()

    types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool]
    sizes = [(batchsize, trajlen, 3),  # True states
             (batchsize, num_particles, 3),  # Initial particles
             (batchsize, None, None, 2),  # Global map: wall, door
             (batchsize, None, None, 3),  # Special maps: empty space map, wall margin map, room type map
             (batchsize, trajlen, 56, 56, 1),  # Depth images, 1 channel
             (batchsize, trajlen, 3),  # Odometry
             (batchsize, trajlen, num_directions),  # Rotation angles
             (batchsize, trajlen, num_directions, 56, 2),  # Points on each wall filter for each direction
             (batchsize,), ]  # is_first_step,

    # turn it into a tf dataset
    def tuplegen():
        for dp in df.get_data():
            yield tuple(dp)

    dataset = tf.data.Dataset.from_generator(tuplegen, tuple(types), tuple(sizes))
    iterator = dataset.make_one_shot_iterator()
    nextdata = iterator.get_next()

    return nextdata, num_samples
