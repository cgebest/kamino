import os
import time
import math
import pickle
import logging

import pandas as pd
import numpy as np
import torch
import scipy.stats
from torch.optim.lr_scheduler import CosineAnnealingLR

from holoclean.dataset import Dataset
from pyvacy.pyvacy.analysis import comp_epsilon
from synthesizer.kamino import hc_repair
from synthesizer.helper import parse_dc, find_sequence, get_relevant_dcs, _check_vio_row, _get_env, _get_num_attrs


class LinearNet(torch.nn.Module):
    def __init__(self, size):
        super(LinearNet, self).__init__()
        self.w = torch.nn.Linear(size, 1)
        self.w.weight.data.fill_(0.)

    def forward(self, n_vios):
        return self.w(n_vios)

    def get_weight(self):
        return self.w.weight.tolist()[0]


def get_preds(path_data, sequence, hitters, paras, all_num_attrs, test_attr=-1):
    """
    Train the model and get predicats on the training data.
    Current implementation trains the models separately from the sampling. Models can be trained once and reused.

    :param path_data path to the csv file of the training dataset
    :param sequence an ordered list of attributes
    :param hitters a set of attributes that participate in at least one of the dcs
    :param all_num_attrs the list of numeric attributes
    :param test_attr debug indicator of how many attributes in the sequence to run
    :param paras parameter dictionary

    Return a list of predications, each for the target attribute from the seq
    """

    cache_name = f"preds_dp{paras['dp']}.pkl"
    try:
        with open(cache_name, 'rb') as input:
            pred_list = pickle.load(input)
            return pred_list
    except (OSError, IOError, FileNotFoundError, EOFError):
        pass

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    tmp_path = f'/tmp/{data_name}.csv'

    df = pd.read_csv(path_data)
    df = df[sequence]
    pred_list = []

    # all_num_attrs = [attr for attr in all_num_attrs if attr not in hitters]
    num_attrs = []

    for attr_idx, attr in enumerate(sequence):
        # debug
        if 0 < test_attr < attr_idx:
            break

        # skip the first
        if attr_idx == 0 or attr not in hitters:
            pred_list.append([])
            continue
        if attr in all_num_attrs:
            num_attrs.append(attr)

        print(f'Generating {attr}')
        # construct a sliced dataframe
        df_slice = df[sequence[:attr_idx + 1]].copy()
        # add a incomplete row at the end of df_slice
        row = df_slice.iloc[0].copy()
        row[attr] = ''
        df_slice = df_slice.append(row, ignore_index=True)
        df_slice.to_csv(tmp_path, index=False)
        # construct the model to predict attr based on previous attributes
        df_pred = hc_repair(data_name, tmp_path, num_attrs, paras, n_val=None)

        df_pred_list = []
        for pred in df_pred:
            if pred['tid'] < df.shape[0] and pred['attribute'] == attr:
                df_pred_list.append(pred)

        # df_pred = df_pred[(df_pred['tid'] < df.shape[0]) & (df_pred['attribute'] == attr)].reset_index(drop=True)

        pred_list.append(pd.DataFrame(df_pred_list))

    try:
        with open(cache_name, 'wb') as output:
            pickle.dump(pred_list, output, pickle.HIGHEST_PROTOCOL)
    except (OSError, IOError, FileNotFoundError, EOFError):
        pass

    return pred_list


def compute_vio_matrix(df, dcs, ds, data_name, sample_n):
    """
    Compute the violation matrix of len(df) * len(dcs).
    m[i][j] represents ith tuple has j violations for jth dc.

    :param df data frame
    :param dcs a list of dcs
    :param an instance of HoloClean dataset
    :param data_name string for table name in postgres
    :param sample_n size of the sample

    Return 2D numpy array
    """

    cache_name = f'vio_matrix_{sample_n}.pkl'
    try:
        with open(cache_name, 'rb') as input:
            vios = pickle.load(input)
            return np.array(vios)
    except (OSError, IOError, FileNotFoundError, EOFError):
        pass

    vios = []

    for tid, row in df.iterrows():
        n_vios = _check_vio_row(df, tid, dcs, ds, data_name)
        vios.append(n_vios)

    try:
        with open(cache_name, 'wb') as output:
            pickle.dump(vios, output, pickle.HIGHEST_PROTOCOL)
    except (OSError, IOError, FileNotFoundError, EOFError):
        pass

    return np.array(vios)


def learn_weights(path_data, path_constraint, paras, sample_n=None):
    """
    Compute the weight of each constraint in path_constraint on path_data
    :param path_data path to the csv file of the dataset.
    :param path_constraint path to the txt file of constraints.
    :param paras parameter dictionary
    :param sample_n sample size

    Return a list, each value corresponds to the weight of an integrity in path_constraint.
    e.g., The first value is the weight of the first constraint.
    """

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]

    df_full = pd.read_csv(path_data)
    if sample_n is None:
        df = df_full
        path_data_sample = path_data
    else:
        df = df_full.dropna(axis=0, inplace=False).sample(n=sample_n, random_state=0)
        df.reset_index(drop=True, inplace=True)
        path_data_sample = f'/tmp/{data_name}_sample{sample_n}.csv'
        df.to_csv(path_data_sample, index=False)

    dcs = parse_dc(path_constraint, list(df.columns))

    paras['N'] = len(df_full)
    paras['n_dcs'] = len(dcs)
    paras['n_cols'] = len(df.columns.tolist())

    all_num_attrs = _get_num_attrs(df, dcs)

    ds = Dataset(name=None, env=_get_env())
    ds.load_data(name=data_name, fpath=path_data_sample, numerical_attrs=all_num_attrs)

    # find the attribute sequence
    sequence, hitters = find_sequence(data_name, dcs, df, rand_sequence=False)
    print(f'Done find the sequence:\n{sequence}\nrelevant attributes= {hitters}')

    start_matrix = time.time()
    vio_matrix = compute_vio_matrix(df, dcs, ds, data_name, sample_n)
    if paras['dp']:
        # add noise to vio_matrix if dp
        sensitivity = np.sqrt(sample_n**2 - sample_n)
        std = np.sqrt(sensitivity * np.log(1.25 / paras['delta_learnW'])) / paras['epsilon_learnW']
        noise = np.random.normal(0., std, (len(df), len(dcs)))
        vio_matrix = np.add(vio_matrix, noise)
        # clip negative value to 0
        vio_matrix[vio_matrix < 0] = 0.
    end_matrix = time.time()
    print(f"Done generating vio_matrix.\tTIME_MATRIX= {end_matrix - start_matrix}")
    logging.info(f'TIME_MATRIX= {end_matrix - start_matrix}')

    # get the predication model
    start_training = time.time()
    pred_list = get_preds(path_data, sequence, hitters, paras, all_num_attrs)
    end_training = time.time()
    logging.info(f'TIME_W_TRAINING= {end_training - start_training}')

    start_weight = time.time()
    # ready to learn the weights
    # each row is used for training once
    n_epoch = 1 + len(df) // paras['minibatch_size']
    n_train = 0

    weight_model = LinearNet(len(dcs))
    optimizer = torch.optim.SGD(weight_model.parameters(), lr=paras['learning_rate'])

    for attr_idx, attr in enumerate(sequence):
        if attr_idx == 0:
            continue
        if attr_idx == len(pred_list):
            break
        print(f'Training for attribute {attr}: {attr_idx + 1} out of {len(sequence)}')
        is_numeric = True if attr in all_num_attrs else False
        # check if attr is in any dc, and that dc has all other attributes in sequence[:attr_idx]
        relevant_dc_idx = get_relevant_dcs(attr, sequence, dcs, exclude_fd=False)
        if len(relevant_dc_idx) == 0:
            print(f'\t skipping {attr} since no dc is covered so far')
            continue

        n_train += 1
        df_pred = pred_list[attr_idx]
        loss = 0
        scheduler = CosineAnnealingLR(optimizer, n_epoch)

        for epoch in range(1, 1 + n_epoch):
            if epoch % 10 == 0:
                print(f'attr= {attr}\tepoch= {epoch}\t{weight_model.get_weight()}')
            print(f'loss= {loss}')
            if paras['dp']:
                optimizer.zero_grad()

            tids = np.where(torch.rand(len(df)) < (paras['minibatch_size'] / len(df)))[0]
            proba_batch = []
            n_vios_batch = []
            for tid_count, tid in enumerate(tids):
                true_val = df[attr][tid]
                # find n_vio for all dcs
                n_vio_list = [0] * len(dcs)
                for dc_idx in relevant_dc_idx:
                    n_vio = vio_matrix[tid][dc_idx]
                    n_vio_list[dc_idx] = n_vio

                n_vios_batch.append(n_vio_list)

                if is_numeric:
                    # compute the probability from guassian pdf
                    sub_df_pred = df_pred[(df_pred['tid'] == tid)]
                    assert sub_df_pred.shape[0] == 1
                    splits = sub_df_pred.loc[sub_df_pred.index[0], 'inferred_val'].split('_')
                    pred_num = float(splits[0])
                    std = float(splits[1])
                    mean = float(splits[2])
                    gaussian = scipy.stats.norm(mean, std)
                    proba = gaussian.pdf(pred_num)
                else:
                    sub_df_pred = df_pred[(df_pred['tid'] == tid) & (df_pred['inferred_val'] == str(true_val).lower())]
                    assert sub_df_pred.shape[0] == 1
                    proba = sub_df_pred.iloc[0]['proba']

                proba_batch.append(proba)

                if tid_count % paras['microbatch_size'] != 0:
                    continue

                n_vios = torch.FloatTensor(n_vios_batch)
                proba = torch.FloatTensor(proba_batch)

                # compute the loss
                y = weight_model(n_vios)
                loss = -1.0 * torch.log(proba * math.e ** y)

                optimizer.zero_grad()

                loss.mean().backward()

                optimizer.step()
                scheduler.step()

                proba_batch = []
                n_vios_batch = []

    end_weight = time.time()
    logging.info(f'TIME_WEIGHT= {end_weight - start_weight}')

    paras['iter_weight'] = n_epoch * n_train

    return weight_model.get_weight()


def toggle_weights(weight_list):
    """
    Set weight to a large number if too small.
    For example, if there is no violation, weight is learned to be 0.

    Return a new list of weights
    """
    threshold = 1e-6
    infinity = 1000.

    new_weight_list = [infinity if w < threshold else w for w in weight_list]
    return new_weight_list


def learn_w_br2000(sample_size, epsilon, delta):
    path_data = './testdata/br2000/br2000.csv'
    path_ic = './testdata/br2000/br2000.ic'

    n_row, n_col = pd.read_csv(path_data).shape
    n_len = len(str(n_row)) + 1

    if sample_size is not None:
        assert epsilon is not None
        assert delta is not None
    else:
        assert epsilon is None
        assert delta is None

    # the DP parameters should be same as the training the models
    # the only privacy cost here is the _learnW
    paras = {
        'reuse_embedding': False,  # set True to reuse the embedding
        'dp': True if epsilon is not None else False,  # set True to enable privacy
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col,
        'epsilon1': .4,  # epsilon1, for adding noise to the
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'minibatch_size': 29,  # batch size to sample for each iteration. default =40
        'microbatch_size': 1,  # micro batch size
        'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
        'learning_rate': 1e-4,
        'iterations': 1500,
        'epsilon_learnW': epsilon,
        'delta_learnW': delta,
    }

    if paras['dp']:

        qs = []
        sigmas = []
        iterations = []

        # DPSGD
        q = paras['minibatch_size'] / paras['n_row']
        qs.append(q)
        sigmas.append(paras['noise_multiplier'])
        iterations.append(paras['iterations'])

        # generator model
        sensitivity = np.sqrt(2)
        std1 = sensitivity * np.sqrt(2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(1):
            qs.append(1.)
            sigmas.append(std1)
            iterations.append(1)

        # learn weights
        sensitivity = np.sqrt(sample_size ** 2 - sample_size)
        std_learnW = sensitivity * np.sqrt(2 * np.log(1.25 / paras['delta_learnW'])) / paras['epsilon_learnW']

        qs.append(sample_size / paras['n_row'])
        sigmas.append(std_learnW)
        iterations.append(1)

        total_epsilon = comp_epsilon(qs, sigmas, iterations, paras['delta'])

        print(total_epsilon)

    weight_list = learn_weights(path_data, path_ic, paras, sample_n=sample_size)
    print(weight_list)

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    dir_name = os.path.dirname(path_data)
    path_weight = f'{dir_name}/{data_name}_sample{sample_size}.w'

    # write to file
    with open(path_weight, 'w') as weight_file:
        weight_file.write(f'{path_data}\n')
        weight_file.write(f'{path_ic}\n')
        for k, v in paras.items():
            weight_file.write(str(k) + '\t' + str(v) + '\n')
        weight_file.write(f'++\n')
        for weight in weight_list:
            weight_file.write(f'{weight}\n')


def learn_w_adult(paras):
    path_data = './testdata/adult/adult.csv'
    path_ic = paras['path_ic']
    sample_size = paras['sample_size']

    weight_list = learn_weights(path_data, path_ic, paras, sample_n=sample_size)
    print(weight_list)

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    dir_name = os.path.dirname(path_data)
    path_weight = f'{dir_name}/{data_name}_sample{sample_size}.w'
    paras['path_weight'] = path_weight

    # write to file
    with open(path_weight, 'w') as weight_file:
        weight_file.write(f'{path_data}\n')
        weight_file.write(f'{path_ic}\n')
        for k, v in paras.items():
            weight_file.write(str(k) + '\t' + str(v) + '\n')
        weight_file.write(f'++\n')
        for weight in weight_list:
            weight_file.write(f'{weight}\n')


if __name__ == '__main__':
    """
    Remove *.pkl from the folder for new dataset
    """
    sample_size = 100
    epsilon = 100.0
    delta = 1e-3
    learn_w_br2000(sample_size, epsilon, delta)

    # sample_size = 100
    # epsilon = 337.0
    # delta = 1e-3
    # learn_w_adult(sample_size, epsilon, delta)
