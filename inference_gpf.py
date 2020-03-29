import numpy as np
import tensorflow as tf
import tqdm

import map_utils as mu
import tf_utils
from arguments import parse_args
from gaussian_utils import GU
from preprocess import get_dataflow

num_directions = 4


def neighbours(x, y):
    """
    :param x:
    :param y:
    :return: list of neighbours (including the point itself) to the current index.
    """
    nb_size = 1
    output = list()
    for i in range(-nb_size, nb_size):
        for j in range(-nb_size, nb_size):
            output.append((x + i, y + j))
    return output


class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, params, global_maps, special_maps):
        """
        :param params: parsed arguments
        :param global_maps, special_maps: tensorflow op (batch, None, None, ch), global maps input.
        Since the map is fixed through the trajectory it can be input to the cell here, instead of
        part of the cell input.
        """
        super(Cell, self).__init__()
        self.params = params
        batch_size = params.batchsize
        num_particles = params.num_particles

        self.gu = GU(params)
        self.trajlen = params.trajlen
        self.batch_size = batch_size
        self.num_particles = num_particles

        self.states_shape = (batch_size, num_particles, 3)
        self.weights_shape = (batch_size, num_particles,)
        self.covariances_shape = (batch_size, num_particles, 3, 3)

        self.global_maps = global_maps
        self.special_maps = special_maps

        wall_map, door_map, gaussian_map = mu.get_maps(global_maps, special_maps)
        self.wall_map = wall_map
        self.door_map = door_map
        self.gaussian_map = gaussian_map

    @property
    def state_size(self):
        return (tf.TensorShape(self.states_shape[1:]),
                tf.TensorShape(self.weights_shape[1:]),
                tf.TensorShape(self.covariances_shape[1:]),)

    @property
    def output_size(self):
        return (tf.TensorShape(self.states_shape[1:]),
                tf.TensorShape(self.weights_shape[1:]),
                tf.TensorShape(self.covariances_shape[1:]))

    def __call__(self, inputs, state, scope=None):
        """
        Implements a particle update.
        :param inputs: observation (batch, 56, 56, ch), odometry (batch, 3), angles (batch, 4),
        filter points (batch, num_directions, 56, 2),
        observation is the sensor reading at time t, odometry is the relative motion from time t to time t+1
        :param state: particle states (batch, K, 3), particle weights (batch, K), particle covariance
        (batch, K, 3, 3); weights are in log space
        :param scope: not used, only kept for the interface. Ops will be created in the current scope.
        :return: outputs, state
        outputs: particle states and weights after the observation update, but before the transition update
        state: updated particle states and weights after both observation and transition updates
        """
        with tf.variable_scope(tf.get_variable_scope()):
            particle_states, particle_weights, particle_covariances = state
            observation, odometry, filter_angles, points = inputs

            def obs_model(particle_states, particle_weights, particle_covariances):
                particle_states, particle_weights, particle_covariances, obs_states, obs_weights, obs_covariances = \
                    self.observation_model(
                        filter_angles, points,
                        particle_states, particle_weights, particle_covariances
                    )
                return particle_states, particle_weights, particle_covariances

            def false_fn(particle_states, particle_weights, particle_covariances):
                return particle_states, particle_weights, particle_covariances

            # Perform observation update depending on whether the input is valid;
            # if invalid, preserve the Gaussians
            particle_states, particle_weights, particle_covariances = tf.cond(
                mu.is_filter_valid(tf.gather(points, 0, axis=0)),
                lambda: obs_model(particle_states, particle_weights, particle_covariances),
                lambda: false_fn(particle_states, particle_weights, particle_covariances)
            )

            # At the first step it outputs the original random states
            # construct output before motion update
            outputs = particle_states, particle_weights, particle_covariances

            # Motion update. this will only affect the particle state input at the next step
            particle_states, particle_covariances = self.transition_model(
                particle_states, particle_covariances, odometry
            )

            # Construct new state
            state = particle_states, particle_weights, particle_covariances

        return outputs, state

    def transition_model(self, states, covariances, odometry):
        """
        Implements a stochastic transition model for localization.
        :param states: tf op (batch, K, 3), particle states before the update.
        :param covariances: tf op (batch, K, 3, 3)
        :param odometry: tf op (batch, 3), odometry reading, relative motion in the robot coordinate frame
        :return: particle_states updated with the odometry and optionally transition noise
        """
        translation_std = self.params.transition_std[0] / self.params.map_pixel_in_meters  # In pixels
        rotation_std = self.params.transition_std[1]  # In radians

        with tf.name_scope('transition'):
            part_x, part_y, part_th = tf.unstack(states, axis=-1, num=3)

            odometry = tf.expand_dims(odometry, axis=1)
            odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

            cos_th = tf.cos(part_th)
            sin_th = tf.sin(part_th)
            delta_x = cos_th * odom_x - sin_th * odom_y
            delta_y = sin_th * odom_x + cos_th * odom_y
            delta_th = odom_th

            new_th = tf.mod(part_th + delta_th + np.pi, 2 * np.pi) - np.pi
            states = tf.stack([part_x + delta_x, part_y + delta_y, new_th], axis=-1)

            pose_cov = tf.square(tf.constant([translation_std, translation_std, rotation_std], tf.float32))
            noise = tf.abs(tf.random_normal(states.get_shape(), mean=0.0, stddev=1.0)) * pose_cov
            noise = tf.matrix_diag(noise)
            covariances = covariances + noise

            return states, covariances

    def observation_model(self, filter_angles, points, particle_states, particle_weights, particle_covariances):
        """
        Apply convolution of the given filter on global maps.
        :param filter_angles: 4 filter rotation angles
        :param points: relative coordinates of points on each filter
        :param observation: camera image
        :return: two sets of Gaussians
        """
        model = self.params.model
        assert model in ['GPF', 'PF']
        if model == 'GPF':
            print('Using GPF')
            obs_states, obs_weights, obs_covariances, conv_maps = self.generate_gaussians(filter_angles, points)

            states, weights, covariances = \
                self.multiply_by_sampling(
                    particle_states, particle_weights, particle_covariances,
                    obs_states, obs_weights, obs_covariances
                )
            return states, weights, covariances, obs_states, obs_weights, obs_covariances
        else:
            print('Using PF')
            points = tf.squeeze(points, axis=0)
            conv_maps = mu.convolve(
                self.wall_map, self.door_map, self.gaussian_map, points
            )

            states, weights = self.update_weights(particle_states, filter_angles, conv_maps)
            covariances = tf.zeros_like(particle_covariances)  # Stub
            return states, weights, covariances, states, weights, covariances

    def update_weights(self, states, filter_angles, conv_maps):
        """
        Observation model used in PF: update weights of given states.
        Assume: batch = 1
        :param states: (batch, num_particles, 3)
        :param filter_angles: (batch, 4)
        :param conv_maps: (num_directions, map_size, map_size)
        :return: states, weights: (batch, num_particles, 3), (batch, num_particles)
        """
        num_particles = tf.shape(states)[1]
        map_size = tf.shape(conv_maps)[1]
        states = tf.reshape(states, (num_particles, 3))
        filter_angles = tf.tile(tf.reshape(filter_angles, (num_directions, 1)), (1, num_particles))

        angles = tf.tile(tf.expand_dims(states[:, 2], axis=0), (num_directions, 1))
        pos = tf.cast(tf.round(states[:, :2]), tf.int32)
        nbs = neighbours(pos[:, 0], pos[:, 1])

        pos_weights = tf.zeros((num_directions, num_particles))
        for nb in nbs:
            pos = nb[1] * map_size + nb[0]
            new_weights = tf.gather(tf.reshape(conv_maps, (num_directions, -1)), pos, axis=1)
            pos_weights = tf.maximum(pos_weights, new_weights)

        ang_weights = -.5 * tf.divide(tf.square(filter_angles - angles), tf.square(self.params.std_theta_obs))
        weights = tf.log(pos_weights) + ang_weights
        weights = tf.reduce_max(weights, axis=0)
        weights = weights - tf.reduce_logsumexp(weights, 0)

        # Resampling step
        indices, weights = tf_utils.sample_k_distinct(num_particles, weights, sample=True, sample_distinct=False)
        states = tf.gather(tf.reshape(states, (num_particles, 3)), indices, axis=0)

        states = tf.expand_dims(states, axis=0)
        weights = tf.expand_dims(weights, axis=0)
        return states, weights

    def generate_gaussians(self, filter_angles, points):
        """
        Apply filters generated from depth image to the global map. Particle states, covariances are used
        as current beliefs.
        :param filter_angles: (batch, 4)
        :param points: (batch, 4, 56, 2)
        :param observation: (batch, 56, 56, ch)
        """
        filter_angles = tf.squeeze(filter_angles, axis=0)
        points = tf.squeeze(points, axis=0)

        conv_maps = mu.convolve(self.wall_map, self.door_map, self.gaussian_map, points)
        states, covariances, weights, labeled = self.gu.gaussians_from_conv_map(conv_maps, filter_angles)

        states = tf.expand_dims(states, axis=0)
        weights = tf.expand_dims(weights, axis=0)
        covariances = tf.expand_dims(covariances, axis=0)
        return states, weights, covariances, conv_maps

    def multiply_mixtures(self, centers_a, weights_a, covariances_a, centers_b, weights_b, covariances_b):
        """
        Full multiplication.
        A: Gaussians representing current beliefs.
        B: Gaussians generated from current observation.
        """
        assert len(centers_a.shape) == len(centers_b.shape) == 3
        assert len(covariances_a.shape) == len(covariances_b.shape) == 4
        assert len(weights_a.shape) == len(weights_b.shape) == 2
        batch_size = centers_a.get_shape().as_list()[0]

        num_particles_a = tf.shape(centers_a)[1]
        num_particles_b = tf.shape(centers_b)[1]
        num_square = num_particles_a * num_particles_b

        # Assumption: diagonal matrices
        covariances_a = tf.matrix_diag_part(covariances_a)
        covariances_b = tf.matrix_diag_part(covariances_b)

        centers_a = tf_utils.repeat(centers_a, num_particles_b)
        centers_b = tf_utils.tile(centers_b, num_particles_a)
        covariances_a = tf_utils.repeat(covariances_a, num_particles_b)
        covariances_b = tf_utils.tile(covariances_b, num_particles_a)

        covariances_sum = covariances_a + covariances_b
        covariances = tf.divide(tf.multiply(covariances_a, covariances_b), covariances_sum)

        centers_diff = (centers_a - centers_b)
        scale_factor = -0.5 * tf.divide(tf.square(centers_diff), covariances_sum)
        scale_factor = tf.reduce_sum(scale_factor, axis=2)
        scale_factor = tf.reshape(scale_factor, (batch_size, num_square))

        centers = tf.divide(tf.multiply(covariances_b, centers_a), covariances_sum) + \
                  tf.divide(tf.multiply(covariances_a, centers_b), covariances_sum)
        centers = tf.reshape(centers, (batch_size, num_square, 3))
        covariances = tf.matrix_diag(covariances)

        # -- Compute outer product of weights, then sample in the rectangular prob dist --
        weights_a = tf_utils.repeat(weights_a, num_particles_b)
        weights_b = tf_utils.tile(weights_b, num_particles_a)
        weights_mul = weights_a + weights_b
        weights_mul = tf.reshape(weights_mul, (batch_size, num_square))
        weights_mul = weights_mul + scale_factor
        weights_mul = weights_mul - tf.reduce_logsumexp(weights_mul, 1, keepdims=True)

        # Reshape back to (batch, num_a, num_b) to sample
        centers = tf.reshape(centers, (batch_size, num_particles_a, num_particles_b, 3))
        weights_mul = tf.reshape(weights_mul, (batch_size, num_particles_a, num_particles_b))
        covariances = tf.reshape(covariances, (batch_size, num_particles_a, num_particles_b, 3, 3))
        return centers, weights_mul, covariances

    def multiply_by_sampling(self, centers_a, weights_a, covariances_a, centers_b, weights_b, covariances_b):
        """
        A: Gaussians representing current beliefs. B: Gaussians generated from current observation.
        The number of particles may be different in A and B.
        :param centers: (batch, num_particles, 3)
        :param weights: (batch, num_particles)
        :param covariances: (batch, num_particles, 3, 3)
        :return:
        """
        assert len(centers_a.shape) == len(centers_b.shape) == 3
        assert len(covariances_a.shape) == len(covariances_b.shape) == 4
        assert len(weights_a.shape) == len(weights_b.shape) == 2
        batch_size = centers_a.get_shape().as_list()[0]
        num_particles = self.num_particles

        centers, weights, covariances = self.multiply_mixtures(
            centers_a, weights_a, covariances_a, centers_b, weights_b, covariances_b
        )

        centers = tf.reshape(centers, (batch_size, -1, 3))
        weights = tf.reshape(weights, (batch_size, -1))
        covariances = tf.reshape(covariances, (batch_size, -1, 3, 3))

        weights = tf.squeeze(weights, axis=0)  # Assumption is batch == 1
        indices, weights = tf_utils.sample_k_distinct(num_particles, weights, sample=False)
        indices, weights = tf.expand_dims(indices, axis=0), tf.expand_dims(weights, axis=0)

        centers = tf.batch_gather(centers, indices)
        covariances = tf.batch_gather(covariances, indices)

        centers = tf.reshape(centers, (batch_size, num_particles, 3))
        weights = tf.reshape(weights, (batch_size, num_particles))
        covariances = tf.reshape(covariances, (batch_size, num_particles, 3, 3))
        return centers, weights, covariances


class Inference(object):
    """ Implements an HMM model to perform inference on robot localization."""

    def __init__(self, params, inputs, labels):
        """
        Calling this will create all tf ops for MGPF.
        :param inputs: inputs to MGPF. Assumed to have the following elements:
        global_maps, init_particle_states, observations, odometries
        :param labels: tf op, labels for training. Assumed to be the true states along the trajectory.
        :param params: parsed arguments
        """
        self.params = params

        # Define ops to be accessed conveniently from outside
        self.outputs = []
        self.hidden_states = []

        self.train_loss_op = None
        self.valid_loss_op = None
        self.all_distance2_op = None
        self.summary_op = None
        self.update_state_op = tf.constant(0)

        # build the network. this will generate the ops defined above
        self.build(inputs, labels)

    def build(self, inputs, labels):
        """
        Unroll the MGPF RNN cell
        """
        self.outputs = self.build_rnn(*inputs)
        self.build_loss_op(self.outputs[0], self.outputs[1], true_states=labels)

    def build_loss_op(self, particle_states, particle_weights, true_states):
        """
        TODO Loss ops are built here, but they are not used.
        Create tf ops for various losses. This should be called only once with is_training=True.
        """
        assert particle_weights.get_shape().ndims == 3

        lin_weights = tf.nn.softmax(particle_weights, dim=-1)

        true_coords = true_states[:, :, :2]
        mean_coords = tf.reduce_sum(tf.multiply(particle_states[:, :, :, :2], lin_weights[:, :, :, None]), axis=2)
        coord_diffs = mean_coords - true_coords

        # Convert from pixel coordinates to meters
        coord_diffs *= self.params.map_pixel_in_meters

        # Coordinate loss component: (x-x')^2 + (y-y')^2
        loss_coords = tf.reduce_sum(tf.square(coord_diffs), axis=2)

        true_orients = true_states[:, :, 2]
        orient_diffs = particle_states[:, :, :, 2] - true_orients[:, :, None]
        # Normalize between -pi..+pi
        orient_diffs = tf.mod(orient_diffs + np.pi, 2 * np.pi) - np.pi
        # Orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        loss_orient = tf.square(tf.reduce_sum(orient_diffs * lin_weights, axis=2))

        # Combine translational and orientation losses
        loss_combined = loss_coords + 0.36 * loss_orient
        loss_pred = tf.reduce_mean(loss_combined, name='prediction_loss')

        self.all_distance2_op = loss_coords
        self.valid_loss_op = loss_pred
        self.train_loss_op = loss_pred

        return loss_pred

    def init_covariances(self, batch_size, num_particles):
        std_pos = self.params.std_pos_max
        std_theta = self.params.std_theta_max
        init_std = tf.constant([std_pos, std_pos, std_theta], tf.float32)
        init_cov = tf.matrix_diag(tf.square(init_std))
        init_cov = tf.expand_dims(tf.expand_dims(init_cov, axis=0), axis=0)
        init_cov = tf.tile(init_cov, (batch_size, num_particles, 1, 1))
        return init_cov

    def build_rnn(self, init_particle_states, global_maps, special_maps,
                  observations, odometries, angles, points, is_first_step):
        """
        Unroll the MGPF RNN cell through time. Input arguments are the inputs to MGPF.
        """
        batch_size = self.params.batchsize
        num_particles = self.params.num_particles
        global_map_ch = global_maps.shape.as_list()[-1]

        init_particle_weights = tf.constant(np.log(1.0 / float(num_particles)),
                                            shape=(batch_size, num_particles), dtype=tf.float32)
        init_particle_covariances = self.init_covariances(batch_size, num_particles)

        # Create hidden state variable
        assert len(self.hidden_states) == 0  # No hidden state should be set before
        self.hidden_states = [
            tf.get_variable('particle_states', shape=init_particle_states.get_shape(),
                            dtype=init_particle_states.dtype, initializer=tf.constant_initializer(0), trainable=False),
            tf.get_variable('particle_weights', shape=init_particle_weights.get_shape(),
                            dtype=init_particle_weights.dtype, initializer=tf.constant_initializer(0), trainable=False),
            tf.get_variable('particle_covariances', shape=init_particle_covariances.get_shape(),
                            dtype=init_particle_covariances.dtype, initializer=tf.constant_initializer(0),
                            trainable=False),
        ]

        # Choose state for the current trajectory segment
        state = tf.cond(tf.reshape(is_first_step, ()),
                        true_fn=lambda: (init_particle_states, init_particle_weights, init_particle_covariances),
                        false_fn=lambda: tuple(self.hidden_states))

        with tf.variable_scope('rnn'):
            # Create variables on GPU
            dummy_cell_func = Cell(
                params=self.params,
                global_maps=tf.zeros((1, 1, 1, global_map_ch), dtype=global_maps.dtype),
                special_maps=tf.zeros((1, 1, 1, 2), dtype=special_maps.dtype),
            )

            dummy_cell_func(
                (tf.zeros([1] + observations.get_shape().as_list()[2:], dtype=observations.dtype),  # observation
                 tf.zeros([1, 3], dtype=odometries.dtype),  # odometry
                 tf.zeros([1, num_directions], dtype=angles.dtype),  # angles
                 tf.zeros([1, num_directions, observations.get_shape().as_list()[2], 2], dtype=points.dtype),
                 # filter_points
                 ),
                (tf.zeros([1, num_particles, 3], dtype=init_particle_states.dtype),  # particle_states
                 tf.zeros([1, num_particles], dtype=init_particle_weights.dtype),  # particle_weights
                 tf.zeros([1, num_particles, 3, 3], dtype=init_particle_covariances.dtype),)  # particle_covariances
            )

            tf.get_variable_scope().reuse_variables()

            # unroll real steps using the variables already created
            cell_func = Cell(
                params=self.params, global_maps=global_maps, special_maps=special_maps
            )

            outputs, state = tf.nn.dynamic_rnn(
                cell=cell_func, inputs=(observations, odometries, angles, points),
                initial_state=state, swap_memory=True, time_major=False,
                parallel_iterations=1, scope=tf.get_variable_scope()
            )

        particle_states, particle_weights, particle_covariances = outputs

        # define an op to update the hidden state, i.e. the particle states and particle weights.
        # this should be evaluated after every input
        with tf.control_dependencies([particle_states, particle_weights, particle_covariances]):
            self.update_state_op = tf.group(
                *(self.hidden_states[i].assign(state[i]) for i in range(len(self.hidden_states)))
            )

        return particle_states, particle_weights, particle_covariances


def inference(params):
    seed = params.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        data, num_samples = get_dataflow(params.trainfiles, params)
        brain = Inference(params=params, inputs=data[1:], labels=data[0])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for step_i in tqdm.tqdm(range(num_samples)):
                out = sess.run([brain.outputs_op])
                tqdm.tqdm.write('Step: {}, Loss: {:.4f}'.format(step_i, 1.))

        except KeyboardInterrupt:
            pass

        except tf.errors.OutOfRangeError:
            print('Data exhausted')

        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    params = parse_args()
    print('params', params)
    inference(params)
