import os

import numpy as np
import tensorflow as tf
import tqdm

from inference_gpf import Inference
from preprocess import get_dataflow

try:
    import ipdb as pdb
except Exception:
    import pdb


def result_file(params):
    """
    Returns a filename for the parameters.
    """
    output = list()
    config = str(params.config)
    output.append(config[config.find('-') + 1:-5])
    output.append(params.model)
    output.append('num_particles_%s' % params.num_particles)
    testfile = params.testfiles[0]
    if 'wallsonly' in testfile:
        output.append('wallsonly')
    if 'tfrecords.0' in testfile:
        output.append('file_%s' % testfile[-3:])
    output.append('start_%s' % 0)
    output.append('std_pos_%s' % int(params.std_pos_max * 10.))
    output.append('std_theta_%s' % int(params.std_theta_max * 10.))
    output = '_'.join(output) + '.out'
    return output


def run_evaluation(params, eval_bools):
    """
    Run evaluation with the parsed arguments.
    :param eval_bools: a boolean array indicating which elements have been evaluated.
    """
    seed = 23
    np.random.seed(seed)
    tf.set_random_seed(seed)

    def write_result(line):
        print(line)
        filename = result_file(params)
        f = open(filename, 'a+')
        f.write(line + '\n')
        f.close()

    def write_params(params):
        filename = result_file(params)
        f = open(filename, 'a+')
        f.write(str(params) + '\n\n')
        f.close()

    # Overwrite for evaluation
    params.batchsize = 1
    write_params(params)

    with tf.Graph().as_default():
        # Test data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            test_data, num_test_samples = get_dataflow(params.testfiles, params)
            test_brain = Inference(params=params, inputs=test_data[1:], labels=test_data[0])

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a session for running Ops on the Graph.
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % int(params.gpu)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            mse_list = []
            success_list = []

            try:
                for step_i in tqdm.tqdm(range(num_test_samples)):
                    if eval_bools[step_i]:
                        # Already evaluated, skipped
                        continue
                    write_result(params.model)
                    all_distances, _ = sess.run([test_brain.all_distance2_op, test_brain.update_state_op])

                    # We have squared differences along the trajectory
                    mse = np.mean(all_distances[0])
                    mse_list.append(mse)
                    all_dist_str = ' '.join([str(d) for d in all_distances])
                    write_result(all_dist_str)

                    # Localization is successful if the RMSE error is below 1m for the last 25% of the trajectory
                    successful = np.all(np.less(all_distances[0][-params.trajlen // 4:], 1.0 ** 2))
                    success_list.append(successful)

                    write_result(params.model)

                    res = 'Step %f, MSE %fcm, Success %f' % (step_i, np.sqrt(mse) * 100, successful)
                    write_result(res)

                    mean_rmse = np.mean(np.sqrt(mse_list))  # a.k.a. MAE
                    total_rmse = np.sqrt(np.mean(mse_list))

                    write_result('Step %f, Mean RMSE (average RMSE per trajectory) = %fcm' % (step_i, mean_rmse * 100))
                    write_result('Step %f, Overall RMSE (reported value) = %fcm' % (step_i, total_rmse * 100))
                    write_result(
                        'Step %f, Success rate = %f%%' % (step_i, np.mean(np.array(success_list, 'i')) * 100))

            except KeyboardInterrupt:
                pass

            except tf.errors.OutOfRangeError:
                print('data exhausted')

            finally:
                coord.request_stop()
            coord.join(threads)

            write_result(params.model)
            mean_rmse = np.mean(np.sqrt(mse_list))
            total_rmse = np.sqrt(np.mean(mse_list))

            write_result('Mean RMSE (average RMSE per trajectory) = %fcm' % (mean_rmse * 100))
            write_result('Overall RMSE (reported value) = %fcm' % (total_rmse * 100))
            write_result('Success rate = %f%%' % (np.mean(np.array(success_list, 'i')) * 100))
