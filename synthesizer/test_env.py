import logging
import time

import pandas as pd
import numpy as np

from pyvacy.pyvacy import analysis
from synthesizer.data_process import preproc_adult, postproc_adult
from synthesizer.util import _analyze_privacy, evaluate_data, copy_log
from synthesizer.kamino import syn_data


def syn_adult100():
    """
    Quick run on the adult data with 100 rows.
    """
    start = time.time()

    path_data = f'./testdata/a100/adult100.csv'
    path_ic = f'./testdata/a100/adult100.ic'

    path_data_preproc = preproc_adult(path_data)

    n_row, n_col = pd.read_csv(path_data_preproc).shape
    n_len = len(str(n_row)) + 1

    paras = {
        'reuse_embedding': True,  # set True to reuse the embedding
        'dp': True,  # set True to enable privacy
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col,  # number of columns in the true dataset
        'epsilon1': .1,  #
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'minibatch_size': 23,
        'microbatch_size': 1,
        'delta': float(f'1e-{n_len}'),
        'learning_rate': 1e-4,
        'iterations': 10
    }

    if paras['dp']:
        epsilon2 = _analyze_privacy(paras)
        paras['epsilon2'] = epsilon2

        gaussian_std = []
        sensitivity = 2
        std1 = np.sqrt(sensitivity * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(1):
            gaussian_std.append(std1)

        epsilon = analysis.epsilon(N=n_row,
                                   batch_size=paras['minibatch_size'],
                                   noise_multiplier=paras['noise_multiplier'],
                                   iterations=paras['iterations'] * (paras['n_col'] - 1),
                                   delta=paras['delta'], gaussian_std=gaussian_std)

        msg = f"epsilon1= {paras['epsilon1']}\t epsilon2= {'{:.2f}'.format(epsilon2)}\t delta= {paras['delta']}\n" \
              f"epsilon= {epsilon}"
        print(msg)

    syn_data(path_data_preproc, path_ic, paras)
    path_syn = paras['path_syn']

    path_data_postproc = postproc_adult(path_syn)

    end = time.time()
    logging.info(f'TIME_WALL= {end - start}')

    evaluate_data(path_data, path_data_postproc, path_ic)
    copy_log(paras)


if __name__ == '__main__':
    """
    The entry point for sanity check
    """

    syn_adult100()

    print('++++++++++++++++++++++\nSUCCESS\n++++++++++++++++++++++')
