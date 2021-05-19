import os
import logging
import time

import pandas as pd
import numpy as np

from pyvacy.pyvacy import analysis
from synthesizer.data_process import  preproc_br2000_separate, postproc_br2000
from synthesizer.util import _analyze_privacy, evaluate_data, copy_log
from synthesizer.ICweight import learn_w_br2000
from synthesizer.kamino import syn_data


def syn_br2000():
    """
    Group binary attributes in the preprocessing step. Restore back from HoloMake process.
    """
    start = time.time()

    path_data = f'./testdata/br2000/br2000.csv'
    path_ic = f'./testdata/br2000/br2000.ic'

    # Group bianry attributes as one attribute. return path includes _concat
    path_data_preproc = preproc_br2000_separate(path_data)

    n_row, n_col = pd.read_csv(path_data_preproc).shape
    n_len = len(str(n_row)) + 1

    paras = {
        'reuse_embedding': True,  # set True to reuse the embedding
        'dp': True,  # set True to enable privacy
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col,  # number of columns in the true data
        'epsilon1': .4,  # epsilon1,
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'minibatch_size': 29,  # batch size to sample for each iteration.
        'microbatch_size': 1,  # micro batch size
        'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
        'learning_rate': 1e-4,
        'iterations': 1500  # =1500 for comparision, =1000 for testing learned weights
    }

    if paras['dp']:
        epsilon2 = _analyze_privacy(paras)

        gaussian_std = []
        sensitivity_hist = 2
        std1 = np.sqrt(sensitivity_hist * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(1):
            gaussian_std.append(std1)

        sensitivity = 100 * 3  # sample size * number of dcs
        std_learnW = np.sqrt(sensitivity * 2. * np.log(1.25 / 1e-3)) / 1.
        gaussian_std.append(std_learnW)

        epsilon = analysis.epsilon(N=n_row,
                                   batch_size=paras['minibatch_size'],
                                   noise_multiplier=paras['noise_multiplier'],
                                   iterations=paras['iterations'] * (paras['n_col'] - 1),
                                   delta=paras['delta'], gaussian_std=gaussian_std)

        paras['epsilon2'] = epsilon2
        msg = f"epsilon1= {paras['epsilon1']}\t epsilon2= {'{:.2f}'.format(epsilon2)}\t delta= {paras['delta']}\n" \
              f"epsilon= {epsilon}"
        print(msg)

    syn_data(path_data_preproc, path_ic, paras)
    path_syn = paras['path_syn']

    path_data_postproc = postproc_br2000(path_syn)

    end = time.time()
    logging.info(f'TIME_WALL= {end - start}')

    evaluate_data(path_data, path_data_postproc, path_ic)
    copy_log(paras)


if __name__ == '__main__':
    """
    The entry point for using kamino to generate synthetic dataset
    """

    sample_size = 100

    # remove cache
    os.system(f'rm *.pkl')
    learn_w_br2000(sample_size, 100, 1e-3)

    # relink to the new learned weight file
    os.system(f'rm ./testdata/br2000/br2000.w')
    os.system(f'ln -s br2000_sample{sample_size}.w ./testdata/br2000/br2000.w')

    syn_br2000()


