import logging
import time
import random

import pandas as pd
import numpy as np

from pyvacy.pyvacy import analysis
from synthesizer.kamino import synthesize_continue
from synthesizer.util import _analyze_privacy, evaluate_data, copy_log


def syn_tpch():
    """
    iid first three attributes, and then use tuple embedding
    For the last two numerical attributes, iid
    """
    start = time.time()

    path_data = f'./testdata/tpch/tpch_order.csv'
    path_data_cat = f'./testdata/tpch/tpch_order_cat.csv'
    path_data_num = f'./testdata/tpch/tpch_order_num.csv'
    path_ic = f'./testdata/tpch/tpch_order.ic'

    # c_custkey, c_mktsegment, c_nationkey,
    # n_name, n_regionkey, o_orderstatus, o_orderpriority,
    # c_acctbal, o_total_price

    df = pd.read_csv(path_data_cat)
    n_row, n_col = df.shape
    n_len = len(str(n_row)) + 1

    paras = {
        'reuse_embedding': True,  # set True to reuse the embedding
        'dp': True,  # set True to enable privacy
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col - 3,  # number of columns in the true dataset
        'epsilon1': .2,  #
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,  #
        'minibatch_size': 19,  #
        'microbatch_size': 1,  # micro batch size
        'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
        'learning_rate': 1e-4,
        'iterations': 2000  # number of iteration. Should be large enough: iterations * minibatch_size > n_row
    }

    std1 = None
    if paras['dp']:
        epsilon2 = _analyze_privacy(paras)
        paras['epsilon2'] = epsilon2

        gaussian_std = []
        sensitivity = 2
        std1 = np.sqrt(sensitivity * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(5):
            gaussian_std.append(std1)

        epsilon = analysis.epsilon(N=n_row,
                                   batch_size=paras['minibatch_size'],
                                   noise_multiplier=paras['noise_multiplier'],
                                   iterations=paras['iterations'] * (paras['n_col'] - 1),
                                   delta=paras['delta'], gaussian_std=gaussian_std)

        msg = f"epsilon1= {5 * paras['epsilon1']}\t epsilon2= {'{:.2f}'.format(epsilon2)}\n" \
              f"epsilon= {epsilon}\t delta= {paras['delta']}"
        print(msg)

    # c_custkey, c_mktsegment, c_nationkey,
    custkeys = list(df['c_custkey'].unique())
    mkesgements = list(df['c_mktsegment'].unique())
    nationkeys = list(df['c_nationkey'].unique())
    probas_custkeys = []
    probas_mkesgements = []
    probas_nationkeys = []

    temp_df_custkeys = df['c_custkey'].value_counts()
    for custkey in custkeys:
        count = temp_df_custkeys[custkey]
        noise = np.random.normal(0, std1)
        noisy_count = max(0, count + noise)
        probas_custkeys.append(noisy_count)
    probas_custkeys = np.array(probas_custkeys) / np.sum(probas_custkeys)

    temp_df_mkesgements = df['c_mktsegment'].value_counts()
    for mkesgement in mkesgements:
        count = temp_df_mkesgements[mkesgement]
        noise = np.random.normal(0, std1)
        noisy_count = max(0, count + noise)
        probas_mkesgements.append(noisy_count)
    probas_mkesgements = np.array(probas_mkesgements) / np.sum(probas_mkesgements)

    temp_df_nationkeys = df['c_nationkey'].value_counts()
    for nationkey in nationkeys:
        count = temp_df_nationkeys[nationkey]
        noise = np.random.normal(0, std1)
        noisy_count = max(0, count + noise)
        probas_nationkeys.append(noisy_count)
    probas_nationkeys = np.array(probas_nationkeys) / np.sum(probas_nationkeys)

    syn_custkeys = []
    syn_mkesgements = []
    syn_nationkeys = []

    while len(syn_custkeys) < df.shape[0]:
        custkey = np.random.choice(custkeys, p=probas_custkeys)

        if custkey in syn_custkeys:
            id = syn_custkeys.index(custkey)
            nationkey = syn_nationkeys[id]
            mkesgement = syn_mkesgements[id]
        else:
            mkesgement = np.random.choice(mkesgements, p=probas_mkesgements)
            nationkey = np.random.choice(nationkeys, p=probas_nationkeys)

        syn_custkeys.append(custkey)
        syn_mkesgements.append(mkesgement)
        syn_nationkeys.append(nationkey)

    dicti = {'c_custkey': syn_custkeys, 'c_mktsegment': syn_mkesgements, 'c_nationkey': syn_nationkeys}
    syn = pd.DataFrame(dicti)

    path_tmp = '/tmp/tpch_3.csv'
    syn.to_csv(path_tmp, index=False)

    synthesize_continue(path_data=path_data_cat, path_syn_old=path_tmp, path_constraint=path_ic,
                        paras=paras, rand_sequence=False, ic_sampling=True)

    path_syn = paras['path_syn']

    ## for numerical attribute
    df_num = _syn_tpch_num(path_data_num, std1)
    df_cat = pd.read_csv(path_syn)

    df_syn = df_cat.copy()
    for num_attr in list(df_num.columns):
        df_syn[num_attr] = df_num[num_attr].copy()

    df_syn.to_csv(path_syn, index=False)

    logging.info(f'path_data_cat= {path_data_cat}\npath_data_num= {path_data_num}\n'
                 f'path_ic= {path_ic}\npath_syn= {path_syn}')
    logging.info(f'{paras}')

    end = time.time()
    time_wall = end - start
    time_train = paras['TIME_TRAINING']
    time_seq = paras['TIME_SEQ']
    time_sampling = time_wall - time_seq * 2 - time_train
    logging.info(f'TIME_WALL= {time_wall}')
    logging.info(f'TIME_SEQ= {time_seq}')
    logging.info(f'TIME_TRAINING= {time_train}')
    logging.info(f'TIME_SAMPLING= {time_sampling}')

    evaluate_data(path_data, path_syn, path_ic)
    copy_log(paras)


def _syn_tpch_num(path_data_num, std1):
    """
    iid two numerical attributes only
    Internal use only
    """
    # c_acctbal, o_totalprice

    df = pd.read_csv(path_data_num)
    n_row, n_col = df.shape

    n_bin = 100

    dicti = {}
    for attr in ['c_acctbal', 'o_totalprice']:
        syn_col = []
        max_acctbal = max(df[attr])
        min_acctbal = min(df[attr])
        bin_width = (max_acctbal - min_acctbal) / n_bin

        df['bin'] = (df[attr] - min_acctbal) // bin_width
        df['bin'] = df['bin'].astype(int)
        df_bin_value_counts = df['bin'].value_counts().sort_index()
        df_bin_value_counts += np.random.normal(0, std1, len(df_bin_value_counts))
        df_bin_value_counts[df_bin_value_counts < 0] = 0

        df_bin_probas = df_bin_value_counts / sum(df_bin_value_counts)

        sampled_bins = np.random.choice(df_bin_value_counts.index, size=n_row, p=df_bin_probas)
        for bin in sampled_bins:
            left = min_acctbal + bin_width * bin
            right = left + bin_width
            v = random.uniform(left, right)
            syn_col.append(v)

        dicti[attr] = syn_col

    syn = pd.DataFrame(dicti)

    path_tmp = '/tmp/tpch_num.csv'
    syn.to_csv(path_tmp, index=False)

    return syn


if __name__ == '__main__':
    """
    The entry point for using kamino to generate synthetic dataset
    """

    syn_tpch()

