import random
import logging

import numpy as np
import scipy.stats
from tqdm import tqdm

from synthesizer.helper import get_relevant_dcs, _get_num_attrs, _get_sampling_paras, _get_weights, Pred, _prep_db, get_n_vio


std_scale = 1.


def update_syn_arsampling(df, syn, index, sequence, df_preds, path_weight, dcs, concat_attr=None):
    """
    Fill the synthetic data value by value using the accept-reject sampling

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
    # limit for the accept-reject sampling
    n_val_limit, n_try = _get_sampling_paras(True)

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

    all_num_attrs = _get_num_attrs(df, dcs)

    # the list to hold all synthetic values for the target attribute
    syn_values = []
    db = None
    count_resampling = 0

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

                pred_vals = []

                if not attr_in_fd:
                    # if this numeric attribute is not in any fds, then sampling
                    samples = np.random.normal(pred_num, (i+1) * std, n_val_limit).tolist()
                    pred_vals += samples
                else:
                    pred_vals = [pred_num]

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
                nprobas = [p / sum(pred_probas) for p in pred_probas]

                # sample 1 value here
                pred_v = np.random.choice(pred_vals, p=nprobas)

                pred_vals = [pred_v]
                pred_probas = [1.]
                nprobas = [1.]

                if len(pred_vals) < n_val_limit:
                    break_flag = True

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

                    r = random.uniform(0, 1)
                    if r <= np.exp(-1. * exp_part):
                        nproba_new = 1.
                    else:
                        nproba_new = 0.

                    nprobas[val_idx] = nproba_new
                # re-normalize
                sum_probas = sum(nprobas)
                if sum_probas > 0:
                    nprobas = [p / sum_probas for p in nprobas]
                    break_flag = True
                else:
                    # none of the predication is clean
                    logging.info(f'none of the predication is clean. tid={_tid}, try {i}')
                    count_resampling += 1
            else:
                break_flag = True

        assert len(pred_vals) > 0
        # if attr participates in FDs, then choose the most likely value
        if attr_in_fd or len(syn_values) == 0:
            index_max = np.argmax(nprobas)
            val = pred_vals[index_max]
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

    logging.info(f'Done update_syn_arsampling. count_resampling= {count_resampling}')

