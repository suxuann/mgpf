"""
Run all experiments in the AISTATS paper.
"""

import itertools
import os
import re

import numpy as np

from arguments import parse_args
from evaluate import run_evaluation, result_file

num_instances = 820


def get_evaluated_booleans():
    """
    Get evaluated boolean arrays for tasks in the directory, so as to resume evaluation.
    Each boolean indicates whether a particular data instance has been evaluated.
    :return:
    """
    fs = os.listdir(os.curdir)
    fs = filter(lambda x: x.endswith('.out'), fs)
    updated = {}

    for filename in fs:
        comb = get_combinations(filename)
        if not comb:
            continue
        if comb not in updated:
            updated[comb] = np.arange(num_instances) * 0.

        scores = get_scores(filename)
        for s in scores:
            index, _, _ = s
            updated[comb][index] = 1
    return updated


def get_combinations(filename):
    """
    Read parameter combinations from the given filename.
    Three parameters: model (GPF or PF), num_particles, furniture (y/n).
    :param filename:
    :return:
    """
    pattern = '(.*)_(G?)PF_num_particles_(\d{1,4})(_wallsonly)?_start_(\d{1,3})' \
              '_std_pos_(\d{1,3})_std_theta_(\d{1,3}).out'
    pattern = re.compile(pattern)
    matches = re.findall(pattern, filename)
    has_match = len(matches) > 0
    if not has_match:
        return None

    matches = list(matches[0])
    matches[1] += 'PF'
    matches[3] = 'wallsonly' if matches[3] else 'furniture'
    output = '_'.join(matches)
    return output


def get_scores(filename):
    """
    Get scores for tracking, localization & global localization.
    """
    f = open(filename, 'r')
    contents = f.read()
    pattern = re.compile(
        'Step (\d{1,3}).000000, MSE (\d{1,5}).(\d{1,6})cm, Success (\d{1}).000000'
    )
    matches = re.findall(pattern, contents)

    def to_score(match):
        index = int(match[0])
        mse = np.square(float('%s.%s' % (match[1], match[2])) / 100.)
        success = int(match[3])
        return index, mse, success

    return map(to_score, matches)


if __name__ == '__main__':
    iters = {
        'model': ['GPF', 'PF'],
        'testfiles': [['./data/wallsonly_test.tfrecords']],
    }

    params = parse_args()
    eval_bool_dict = get_evaluated_booleans()
    print(params)

    # Vary the number of particles
    task = params.config[params.config.find('-') + 1:-5]
    assert task in ['tracking', 'localization', 'global', 'tworoom']
    if task == 'tracking':
        iters['num_particles'] = [50, 100, 300]
    else:
        iters['num_particles'] = [100, 300, 600]

    iter_values = []
    for v in iters.values():
        iter_values.append(v)

    count = 0
    for t in itertools.product(*iter_values):
        print(count, t)
        count += 1
        params.model = t[2]
        params.num_particles = t[1]
        params.testfiles = t[0]

        # Read from result file to resume evaluation
        filename = result_file(params)
        comb = get_combinations(filename)
        comb_bools = eval_bool_dict.get(comb, np.arange(num_instances) * 0.)
        run_evaluation(params, comb_bools)
