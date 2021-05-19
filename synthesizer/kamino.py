import os
import time
import socket
from datetime import datetime
import logging
import pickle
import re

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.stats
from tqdm import tqdm

import holoclean
from holoclean.detect import NullDetector
from holoclean.repair import EmbeddingFeaturizer
from synthesizer.helper import parse_dc, find_sequence, get_relevant_dcs, _get_num_attrs, _get_sampling_paras, \
    _get_weights, Pred, _prep_db, get_n_vio
from synthesizer.hm4AR import update_syn_arsampling
from synthesizer.hm4mcmc import mcmc
from synthesizer.metadata import _get_num_attrs_quant
from synthesizer.util import get_tpch_only

tbl_name = 'syn'
std_scale = 1.
concat_threshold = 2


def hc_repair(data_name, tmp_path, num_attrs, paras, attr=None, n_val=-1):
    """
    Filling value by value in the synthetic dataset

    :param data_name string for the true dataset
    :param tmp_path temporary path for partial data
    :param num_attrs a list of numeric attributes
    :param paras parameters
    :param n_val number of value to predict. Set None to predicate entire domain

    Return a list of dumped predicates from autoencoder
    """

    m = paras.get('MCMC', 0)

    path_preds = None
    if m > 0:
        assert attr is not None
        # load the model from previous save, without training
        dir_preds = os.path.abspath(f"./_models")
        os.makedirs(dir_preds, exist_ok=True)
        path_preds = f"{dir_preds}/model_{data_name}_{attr}.pkl"

        if os.path.exists(path_preds):
            with open(path_preds, 'rb') as input:
                df_preds = pickle.load(input)

            logging.info('DONE with loading model from file')
            return df_preds

    if n_val is not None:
        # n_val is None for weight learning
        # train the model in the standard way, or the saved model does not exist for mcmc
        n_val_limit, n_try = _get_sampling_paras(paras.get('AR', False))
        n_val = n_val_limit * n_try

    hc = holoclean.HoloClean(db_name='db4kamino', domain_thresh_1=0.0, domain_thresh_2=0.0, max_domain=10000,
                             cor_strength=0, weight_decay=0., learning_rate=0.001, threads=1, batch_size=1,
                             verbose=False, timeout=3 * 60000, infer_mode='dk',
                             privacy=paras['dp'], delta=paras['delta'], iterations=paras['iterations'],
                             noise_multiplier=paras['noise_multiplier'], l2_norm_clip=paras['l2_norm_clip'],
                             minibatch_size=paras['minibatch_size'], microbatch_size=paras['microbatch_size']).session

    hc.load_data(data_name, tmp_path, numerical_attrs=num_attrs)

    detectors = [NullDetector()]
    hc.detect_errors(detectors)

    num_attr_groups = []
    quantized_num = []
    num_attrs_quant = _get_num_attrs_quant(data_name)

    for num_attr in num_attrs:
        num_attr_groups.append([num_attr])

        if num_attrs_quant is not None and num_attr in num_attrs_quant:
            quantized_num.append((num_attrs_quant[num_attr], [num_attr]))

    hc.quantize_numericals(quantized_num)
    hc.generate_domain()

    embedfest = EmbeddingFeaturizer(reuse_embedding=paras['reuse_embedding'],
                                    numerical_attr_groups=num_attr_groups)
    embedfest.setup_featurizer(hc.env, hc.ds)

    if m > 0:
        assert attr is not None and path_preds is not None
        df_preds = embedfest.embedding_model.dump_predictions_hm(n_val_limit=n_val, include_std=True, fpath=path_preds)
    else:
        df_preds = embedfest.embedding_model.gen_predictions(n_val_limit=n_val, include_std=True)

    logging.info('DONE with training the autoencoder model')

    return df_preds


def update_syn_hm(df, syn, index, sequence, df_preds, path_weight, dcs, concat_attr=None):
    """
    Fill the synthetic data value by value

    :param df dataframe true dataset
    :param syn dataframe synthetic dataset
    :param index the index of target attribute
    :param sequence the schema sequence
    :param df_preds the output from the model
    :param path_weight file path to constraint weight. If None, do not use weight sampling
    :param dcs a list of dcs
    :param concat_attr the name of concat attributes

    Update the syn dataframe in place
    """

    # limit for normal run
    n_val_limit, n_try = _get_sampling_paras(False)

    attr = sequence[index]

    weights = []
    relevant_dc_idx = []
    attr_in_fd = False
    if path_weight is not None:
        weights = _get_weights(path_weight)
        relevant_dc_idx = get_relevant_dcs(attr, sequence, dcs, exclude_fd=True)
        relevant_dc_idx_all = get_relevant_dcs(attr, sequence, dcs, exclude_fd=False)

        logging.info(f"relevant_dc_id= {relevant_dc_idx}\nrelevant_dc_idx_all= {relevant_dc_idx_all}")

        if len(relevant_dc_idx_all) != len(relevant_dc_idx):
            attr_in_fd = True

        # enforce fd by enabling below line
        relevant_dc_idx = relevant_dc_idx_all

        for w_idx, w in enumerate(weights):
            if w_idx not in relevant_dc_idx:
                weights[w_idx] = 0
    weights = np.asarray(weights)
    # assert np.count_nonzero(weights) == len(relevant_dc_idx)

    all_num_attrs = _get_num_attrs(df, dcs)

    # the list to hold all synthetic values for the target attribute
    syn_values = []
    db = None

    # create the df_preds generator
    if concat_attr is None:
        df_preds_gen = iter(Pred(df_preds, attr, len(df)))
    else:
        df_preds_gen = iter(Pred(df_preds, concat_attr, len(df)))

    for _tid, row in tqdm(syn.iterrows(), total=syn.shape[0]):
        tid = _tid + len(df)

        pred_tid, pred_vals_all, pred_probas_all = next(df_preds_gen)
        assert tid == pred_tid

        break_flag = False
        for i in range(n_try):
            if break_flag:
                break

            if len(pred_probas_all) == 1 and pred_probas_all[0] < 0:
                # numerical attribute, sample from normal distribution
                splits = pred_vals_all[0].split('_')
                pred_num = float(splits[0])
                std = float(splits[1]) * std_scale
                pred_vals = [pred_num]

                if not attr_in_fd:
                    # if this numeric attribute is not in any fds, then sampling
                    samples = np.random.normal(pred_num, (i+1) * std, n_val_limit).tolist()
                    pred_vals += samples

                gaussian = scipy.stats.norm(pred_num, (i+1) * std)
                pred_probas = [gaussian.pdf(v) for v in pred_vals]
                nprobas = [p / sum(pred_probas) for p in pred_probas]

                # clip to domain, assuming the active domain from the true data
                min_v = min(df[attr])
                max_v = max(df[attr])
                pred_vals = [max(min(x, max_v), min_v) for x in pred_vals]

            else:
                # categorical attribute
                sorted_pair = sorted(zip(pred_probas_all, pred_vals_all), reverse=True)
                pred_vals = [x for _, x in sorted_pair]
                pred_probas = [x for x, _ in sorted_pair]

                pred_vals = pred_vals[i * n_val_limit: (i+1) * n_val_limit]
                pred_probas = pred_probas[i * n_val_limit: (i+1) * n_val_limit]

                if len(pred_vals) < n_val_limit:
                    break_flag = True

                nprobas = [p / sum(pred_probas) for p in pred_probas]

            if path_weight is not None and len(relevant_dc_idx) > 0 and len(pred_vals) > 1 and len(syn_values) > 0:
                if db is None:
                    db = _prep_db(syn, syn_values, attr, df[attr].dtype, dcs, relevant_dc_idx)
                # get the number of violations if setting each value from the predication.
                # set true_count=False for faster. Return a marix of size len(pred_vals) X len(dcs)
                n_vios = get_n_vio(syn, syn_values, attr, pred_vals, dcs, relevant_dc_idx, all_num_attrs, db, True)

                for val_idx, pred_val in enumerate(pred_vals):
                    # check #vio if assigning the cell with pred_val
                    n_vios_val = n_vios[val_idx, :]
                    # adjust the nprobas based on n_vio
                    exp_part = sum(weights * n_vios_val)
                    nproba_new = nprobas[val_idx] * np.exp(-1. * exp_part)
                    nprobas[val_idx] = nproba_new
                # re-normalize
                sum_probas = sum(nprobas)
                if sum_probas > 0:
                    nprobas = [p / sum_probas for p in nprobas]
                    break_flag = True
                else:
                    # none of the predication is clean
                    logging.info(f'none of the predication is clean. tid={_tid}, try {i}')
                    nprobas = [p / sum(pred_probas) for p in pred_probas]
                    pred_vals = pred_vals[:len(nprobas)]
            else:
                break_flag = True

        assert len(pred_vals) > 0
        # if attr participates in FDs, then choose the most likely value
        if attr_in_fd or len(syn_values) == 0:
            index_max = np.argmax(nprobas)
            val = pred_vals[index_max]
            # val = pred_vals[0]
        else:
            val = np.random.choice(pred_vals, p=nprobas)
        syn_values.append(val)

    if db is not None:
        db.engine.dispose()

    assert len(syn_values) == len(syn)

    if concat_attr is None:
        syn[attr] = syn_values
        syn[attr] = syn[attr].astype(df[attr].dtype)
    else:
        syn[concat_attr] = syn_values


def create_missing_values(target_attr, df, syn_o, tmp_path):
    """
    Expand the dataframe with one more attribute, with empty values

    :param target_attr the target attribute
    :param df dataframe of true dataset
    :param syn_o dataframe on which we expand
    :param tmp_path path to store the expanded dataframe
    """
    syn = syn_o.copy()
    syn[target_attr] = ''
    train_df = df[syn.columns]
    train_df.to_csv(tmp_path, index=False)

    with open(tmp_path, 'a+') as file:
        syn.to_csv(file, header=False, index=False)


def gen_iid_attr(attr, df, is_numeric, paras, epsilon=None):
    """
    Generate the attr based on the distribution in df

    :param attr target attribute
    :param df dataframe holding the true distribution of attr
    :param is_numeric boolean value
    :param paras parameters
    :param epsilon epsilon of dp

    Return a dataframe of synthetic attr, of equal length to df
    """
    dicti = {}
    if is_numeric:
        # use GMM
        if epsilon is None:
            gmm = GaussianMixture(n_components=5).fit(np.array(df[attr]).reshape(-1, 1))
            Xnew = gmm.sample(df.shape[0])
            iid = Xnew[0].reshape(1, -1)[0]
        else:
            # todo: add noise to gmm
            return None
    else:
        # categorical attribute
        n_domain = len(df[attr].unique())
        if epsilon is None:
            noise = np.zeros(n_domain)
        else:
            # noise = np.random.laplace(0, 1.0 / epsilon, n_domain)
            sensitivity = 2
            std1 = np.sqrt(sensitivity * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
            noise = np.random.normal(0, std1)

        temp_df = df[attr].value_counts() + noise

        temp_df = abs(temp_df)
        temp_df_sum = sum(temp_df)
        temp_df /= temp_df_sum
        values = temp_df.index.values.tolist()
        probs = temp_df.to_list()
        iid = np.random.choice(values, size=df.shape[0], p=probs)

    dicti[attr] = iid
    re = pd.DataFrame(dicti)
    re[attr] = re[attr].astype(df[attr].dtype)
    assert len(re) == len(df)

    return re


def synthesize(path_data, path_constraint, paras, rand_sequence, ic_sampling, concat):
    """
    The main function of kamino process. The synthetic data is stored within the same directory

    :param path_data path to the true dataset
    :param path_constraint path to the ics
    :param paras env dict
    :param rand_sequence boolean value to choose random sequence, or sequence based on constraints
    :param ic_sampling boolean value to enable sampling based on constraints
    :param concat boolean value to concat attributes during the process
    """
    start_synthesize = time.time()
    start_time = datetime.now()
    run_ts = start_time.strftime("%m%d-%H%M")
    hostname = socket.gethostname()

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path_data)

    tmp_path = f'/tmp/{data_name}_{run_ts}.csv'
    dir_syn = f'{dir_path}/{run_ts}_{hostname}/'
    os.makedirs(dir_syn, exist_ok=True)

    if paras['dp']:
        epsilon_str = str('{:.2f}'.format(float(paras['epsilon1']) + float(paras['epsilon2'])))
    else:
        epsilon_str = 'INF'

    m = paras.get("MCMC", 0)
    syn_file = f"{data_name}_dp{epsilon_str}_rand{rand_sequence}_ic{ic_sampling}_m{str(m)}.syn"
    path_syn = f"{dir_syn}/{syn_file}"
    paras['path_syn'] = path_syn
    paras['dir_syn'] = dir_syn
    paras['data_name'] = data_name
    paras['synfile_name'] = syn_file

    if ic_sampling:
        sdata_name = re.sub("\_concat", '', data_name)
        path_weight = f'{dir_path}/{sdata_name}.w'
        assert os.path.isfile(path_weight)
        logging.info(f"Path_weight= {path_weight}")
    else:
        path_weight = None

    df = pd.read_csv(path_data)
    df.dropna(axis=0, inplace=True)
    dcs = parse_dc(path_constraint, list(df.columns))

    all_num_attrs = _get_num_attrs(df, dcs)
    logging.info(f'all_num_attrs= {all_num_attrs}')

    start_seq = time.time()
    # find the attribute sequence
    sequence, _ = find_sequence(data_name, dcs, df, rand_sequence)

    while sequence[0] in all_num_attrs:
        sequence, _ = find_sequence(data_name, dcs, df, rand_sequence)

    concat_attrs = []
    if concat:
        for attr in sequence:
            if df[attr].nunique() <= concat_threshold:
                concat_attrs.append(attr)

    end_seq = time.time()
    logging.info(f'sequence= {sequence}\nconcat_attrs= {concat_attrs}')
    logging.info(f'TIME_SEQ= {end_seq - start_seq}')
    paras['TIME_SEQ'] = paras.get('TIME_SEQ', 0) + end_seq - start_seq

    syned_attrs = []
    syned_num_attrs = []

    start_iid = time.time()
    # initialize the first attribute
    attr = sequence[0]
    if concat:
        # todo: assuming the first attribute is not in the group of concat attributes
        assert attr not in concat_attrs
    epsilon1 = paras['epsilon1'] if paras['dp'] else None
    syn = gen_iid_attr(attr, df, attr in all_num_attrs, paras, epsilon1)
    syned_attrs.append(attr)

    end_iid = time.time()
    logging.info(f'TIME_IID= {end_iid - start_iid}')

    time_training = 0
    time_sampling = 0
    max_time_training = 0
    embedding_flag = False
    last_embedding = False
    ar = paras.get('AR', False)
    m = paras.get('MCMC', 0)

    # loop over the sequence
    # for i in range(1, 2):
    for i in range(1, len(sequence)):
        attr = sequence[i]

        logging.info(f"existing attrs= {syn.columns.to_list()}\ntarget_attr= {attr}")

        if concat and attr in concat_attrs:
            attr_concat_idx = concat_attrs.index(attr)
            if attr_concat_idx < len(concat_attrs) - 1:
                logging.info(f"continue to next due to concat")
                continue
            else:
                # the last attribute to concat
                logging.info(f"begin to concat")
                start_train = time.time()
                concat_attr = '_'.join(concat_attrs)
                df_copy = df.copy()
                df_copy[concat_attr] = df_copy[concat_attrs].agg('_'.join, axis=1)
                df_copy.drop(columns=concat_attrs, inplace=True)

                create_missing_values(concat_attr, df_copy, syn, tmp_path)
                df_preds = hc_repair(data_name, tmp_path, syned_num_attrs, paras, concat_attr)

                end_train = time.time()
                crnt_training_time = end_train - start_train
                max_time_training = max(max_time_training, crnt_training_time)
                time_training += crnt_training_time

                start_sampling = time.time()

                if ar:
                    update_syn_arsampling(df, syn, i, sequence, df_preds, path_weight, dcs, concat_attr=concat_attr)
                else:
                    update_syn_hm(df, syn, i, sequence, df_preds, path_weight, dcs, concat_attr=concat_attr)

                if m > 0:
                    mcmc(df, syn, i, sequence, df_preds, path_weight, dcs, concat_attr=concat_attr, m=m)

                end_sampling = time.time()
                time_sampling += end_sampling - start_sampling

                syn[concat_attrs] = syn[concat_attr].str.split("_", expand=True)
                syn.drop(columns=[concat_attr], inplace=True)
                syned_attrs.extend(concat_attrs)

                embedding_flag = True
                continue

        if attr in all_num_attrs:
            syned_num_attrs.append(attr)

        syned_attrs.append(attr)

        start_train = time.time()
        create_missing_values(attr, df, syn, tmp_path)

        # need to disable reuse_embedding immediately after concat
        if embedding_flag and paras['reuse_embedding']:
            last_embedding = True
            paras['reuse_embedding'] = False

        df_preds = hc_repair(data_name, tmp_path, syned_num_attrs, paras, attr)
        end_train = time.time()
        crnt_training_time = end_train - start_train
        max_time_training = max(max_time_training, crnt_training_time)
        time_training += crnt_training_time

        if embedding_flag and last_embedding:
            paras['reuse_embedding'] = True
            embedding_flag = False

        start_sampling = time.time()
        if ar:
            update_syn_arsampling(df, syn, i, sequence, df_preds, path_weight, dcs)
        else:
            update_syn_hm(df, syn, i, sequence, df_preds, path_weight, dcs)

        if m > 0:
            mcmc(df, syn, i, sequence, df_preds, path_weight, dcs, m=m)
        print(f'target_attr= {attr}')

        end_sampling = time.time()
        time_sampling += end_sampling - start_sampling

    # reorder the attributes per the true dataset
    syn = syn[df.columns]
    syn.to_csv(path_syn, header=True, index=False)

    end_synthesize = time.time()
    logging.info(f'TIME_TRAINING= {time_training}')
    logging.info(f'TIME_TRAINING_MAX= {max_time_training}')
    logging.info(f'TIME_SAMPLING= {time_sampling}')
    logging.info(f'TIME_ALL= {end_synthesize - start_synthesize}')
    paras['TIME_TRAINING'] = paras.get('TIME_TRAINING', 0) + time_training


def syn_data(path_data, path_ic, paras):
    synthesize(path_data=path_data, path_constraint=path_ic, paras=paras, concat=False,
               rand_sequence=False, ic_sampling=True)

    path_syn = paras['path_syn']
    logging.info(f'path_data= {path_data}\npath_ic= {path_ic}\npath_syn= {path_syn}')
    logging.info(f'{paras}')


#################################################################
# OPTIMIZATION METHODS BELOW
#################################################################
def synthesize_continue(path_data, path_syn_old, path_constraint, paras, rand_sequence, ic_sampling):

    start_time = datetime.now()
    run_ts = start_time.strftime("%m%d-%H%M")
    hostname = socket.gethostname()

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path_data)
    syndata_name = os.path.splitext(os.path.basename(path_syn_old))[0]

    tmp_path = f'/tmp/{data_name}_{run_ts}.csv'
    dir_syn = f'{dir_path}/{run_ts}_{hostname}/'
    os.makedirs(dir_syn, exist_ok=True)

    if paras['dp']:
        epsilon_str = str('{:.2f}'.format(float(paras['epsilon1']) + float(paras['epsilon2'])))
    else:
        epsilon_str = 'INF'

    syn_file = f"{data_name}_dp{epsilon_str}_rand{rand_sequence}_ic{ic_sampling}.syn"
    path_syn = f"{dir_syn}/{syn_file}"
    paras['path_syn'] = path_syn
    paras['dir_syn'] = dir_syn
    paras['data_name'] = data_name
    paras['synfile_name'] = syn_file

    if ic_sampling:
        path_weight = f'{dir_path}/{data_name}.w'
        print(f'path_weight= {path_weight}')
        assert os.path.isfile(path_weight)
    else:
        path_weight = None

    df = pd.read_csv(path_data)
    df.dropna(axis=0, inplace=True)
    dcs = parse_dc(path_constraint, list(df.columns))

    # find the attribute sequence
    start_seq = time.time()
    sequence, _ = find_sequence(syndata_name, dcs, df, rand_sequence)
    end_seq = time.time()
    time_seq = end_seq - start_seq
    logging.info(f'sequence= {sequence}')

    if len(sequence) != df.shape[1]:
        combo_attrs = []
        for attr in df.columns:
            if attr not in sequence:
                combo_attrs.append(attr)

        df['combo'] = df[combo_attrs].agg('_'.join, axis=1)
        df.drop(columns=combo_attrs, inplace=True)

    all_num_attrs = _get_num_attrs(df, dcs)

    syn = pd.read_csv(path_syn_old)

    syned_attrs = list(syn.columns)
    syned_num_attrs = []
    for a in syned_attrs:
        if a in all_num_attrs:
            syned_num_attrs.append(a)

    # loop over the sequence
    time_train = 0
    start_idx = len(syned_attrs)
    for i in range(start_idx, len(sequence)):
        attr = sequence[i]

        logging.info(f"existing attrs= {syn.columns.to_list()}\ntarget_attr= {attr}")

        if attr in all_num_attrs:
            syned_num_attrs.append(attr)

        syned_attrs.append(attr)

        create_missing_values(attr, df, syn, tmp_path)

        if paras['reuse_embedding']:
            if i == start_idx:
                paras['reuse_embedding'] = False
            else:
                paras['reuse_embedding'] = True

        start_train = time.time()
        df_preds = hc_repair(data_name, tmp_path, syned_num_attrs, paras)
        end_train = time.time()
        time_train += end_train - start_train

        if get_tpch_only():
            update_syn = update_syn_tpch
        else:
            update_syn = update_syn_hm

        update_syn(df, syn, i, sequence, df_preds, path_weight, dcs)

    # reorder the attributes per the true dataset
    syn = syn[df.columns]
    syn.to_csv(path_syn, header=True, index=False)

    paras['TIME_TRAINING'] = paras.get('TIME_TRAINING', 0) + time_train
    paras['TIME_SEQ'] = paras.get('TIME_SEQ', 0) + time_seq


def update_syn_tpch(df, syn, index, sequence, df_preds, path_weight, dcs, concat_attr=None):
    """
    FOR TPC-H ONLY, taking hard FDs
    """

    assert get_tpch_only() is True

    # limit for normal run
    n_val_limit, n_try = _get_sampling_paras(False)

    attr = sequence[index]

    relevant_dc_idx = []
    relevant_attrs = set()
    attr_in_fd = False
    if path_weight is not None:
        relevant_dc_idx = get_relevant_dcs(attr, sequence, dcs, exclude_fd=True)
        relevant_dc_idx_all = get_relevant_dcs(attr, sequence, dcs, exclude_fd=False)

        logging.info(f"relevant_dc_id= {relevant_dc_idx}\nrelevant_dc_idx_all= {relevant_dc_idx_all}")

        if len(relevant_dc_idx_all) != len(relevant_dc_idx):
            attr_in_fd = True
            for x in relevant_dc_idx_all:
                for a in dcs[x].components:
                    relevant_attrs.add(a)
            relevant_attrs.remove(attr)
            relevant_attrs = list(relevant_attrs)

        # enforce fd by enabling below line
        relevant_dc_idx = relevant_dc_idx_all

    # assert np.count_nonzero(weights) == len(relevant_dc_idx)

    # the list to hold all synthetic values for the target attribute
    syn_values = []

    # create the df_preds generator
    if concat_attr is None:
        df_preds_gen = iter(Pred(df_preds, attr, len(df)))
    else:
        df_preds_gen = iter(Pred(df_preds, concat_attr, len(df)))

    tpch_dict = {}

    for _tid, row in tqdm(syn.iterrows(), total=syn.shape[0]):
        tid = _tid + len(df)

        pred_tid, pred_vals_all, pred_probas_all = next(df_preds_gen)
        assert tid == pred_tid

        break_flag = False
        for i in range(n_try):
            if break_flag:
                break

            if len(pred_probas_all) == 1 and pred_probas_all[0] < 0:
                # numerical attribute, sample from normal distribution
                logging.error("ERROR: TPC-H 1m should not have numerical attributes.")
            else:
                # categorical attribute
                sorted_pair = sorted(zip(pred_probas_all, pred_vals_all), reverse=True)
                pred_vals = [x for _, x in sorted_pair]
                pred_probas = [x for x, _ in sorted_pair]

                pred_vals = pred_vals[i * n_val_limit: (i+1) * n_val_limit]
                pred_probas = pred_probas[i * n_val_limit: (i+1) * n_val_limit]

                if len(pred_vals) < n_val_limit:
                    break_flag = True

                nprobas = [p / sum(pred_probas) for p in pred_probas]

            if path_weight is not None and len(relevant_dc_idx) > 0 and len(pred_vals) > 1:

                # FIRST TO CHECK IF ANY PREVIOUS VLAUE CAN BE REUSED
                key = ""
                for ra in relevant_attrs:
                    key += '_' + row[ra]

                if key in tpch_dict:
                    previous_val = tpch_dict[key]
                    pred_vals = [previous_val]
                else:
                    index_max = np.argmax(nprobas)
                    val = pred_vals[index_max]
                    pred_vals = [val]
                    tpch_dict[key] = val

                nprobas = [1.]
                break_flag = True

        assert len(pred_vals) > 0
        # if attr participates in FDs, then choose the most likely value
        if attr_in_fd or len(syn_values) == 0:
            index_max = np.argmax(nprobas)
            val = pred_vals[index_max]
        else:
            sum_probas = sum(nprobas)
            assert sum_probas > 0
            nprobas = [p / sum_probas for p in nprobas]
            val = np.random.choice(pred_vals, p=nprobas)
        syn_values.append(val)

    assert len(syn_values) == len(syn)

    if concat_attr is None:
        syn[attr] = syn_values
        syn[attr] = syn[attr].astype(df[attr].dtype)
    else:
        syn[concat_attr] = syn_values


