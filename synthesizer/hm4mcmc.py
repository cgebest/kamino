import random
from string import Template
import logging

import pandas as pd
import numpy as np
import scipy.stats
from tqdm import tqdm

from holoclean.dataset import DBengine
from synthesizer.helper import get_relevant_dcs, _get_env, _get_num_attrs, _get_sampling_paras, _update_cnf, _get_weights


tbl_name = 'syn'
std_scale = 1.
concat_threshold = 2


class PredMCMC:

    def __init__(self, df_preds, attr, start_tid):
        self.attr = attr
        self.start_tid = start_tid
        df = pd.DataFrame(df_preds)
        df.drop(df[(df.tid < start_tid) | (df.attribute != attr)].index, inplace=True)
        self.df = df

    def get_preds(self, tid):
        tid_all = tid + self.start_tid
        crnt_df = self.df[self.df['tid'] == tid_all]
        pred_vals = crnt_df['inferred_val'].tolist()
        pred_probas = crnt_df['proba'].tolist()

        return tid_all, pred_vals, pred_probas


def _prep_db_mcmc(syn, dcs, relevant_dc_idx):
    """
    Create the synthetic table, which will be used to get the number of violations for MCMC resampling
    """

    # create a db engine
    env = _get_env()
    db = DBengine(env['db_user'], env['db_pwd'], env['db_name'], env['db_host'],
                  pool_size=env['threads'], timeout=env['timeout'])

    syn.to_sql(tbl_name, con=db.engine, if_exists='replace', index=True, index_label='tid')

    # find relevant attributes and create index on them
    index_attrs = set()
    for dc_idx, dc in enumerate(dcs):
        if relevant_dc_idx is not None and dc_idx not in relevant_dc_idx:
            continue
        for attr in dc.components:
            index_attrs.add(attr)
    for index_attr in index_attrs:
        dml = f'CREATE INDEX "idx_{index_attr}" ON {tbl_name}("{index_attr}")'
        db.execute_query_no_return(dml)

    logging.info('DONE with db preparation')
    return db


sql2_template_count = Template('select count(*) from "$tbl_name" t1 where t1."tid" != $tid and $cnf')


def get_n_vio_mcmc(syn, attr, tid, vals, dcs, relevant_dc_idx, all_num_attrs, db):
    """
    Compute the number of violations if assigning attr with val

    Return 2D list of size len(vals) X len(dcs). Each row corresponds the violations of a potential value.
    """
    row = syn.iloc[tid].copy()

    n_vios = np.zeros((len(vals), len(dcs)), dtype=int)
    queries = []
    sql_idx_to_matrix_idx = {}

    for val_idx, val in enumerate(vals):
        for dc_idx, dc in enumerate(dcs):
            if relevant_dc_idx is not None and dc_idx not in relevant_dc_idx:
                continue
            # for this val, and on this particular dc
            row[attr] = val
            attr_vals = [row[attr] for attr in dc.components]
            if np.nan in attr_vals:
                continue

            if len(dc.tuple_names) == 2:
                replaced_cnf = _update_cnf(dc.cnf_form, row, all_num_attrs)
                sql = sql2_template_count.substitute(tbl_name=tbl_name, tid=tid, cnf=replaced_cnf)
            elif len(dc.tuple_names) == 1:
                sql = None
            else:
                sql = None

            sql_idx_to_matrix_idx[len(queries)] = (val_idx, dc_idx)
            queries.append(sql)

            re = db.execute_query(sql)
            n_vios[val_idx, dc_idx] = re[0][0]

    return n_vios


def mcmc(df, syn, index, sequence, df_preds, path_weight, dcs, concat_attr=None, m=0):
    """
    MCMC resampling

    :param df dataframe true dataset
    :param syn dataframe synthetic dataset
    :param index the index of target attribute
    :param sequence the schema sequence
    :param df_preds the output from the model
    :param path_weight file path to constraint weight. If None, do not use weight sampling
    :param dcs a list of dcs
    :param concat_attr the name of concat attributes
    :param m number of steps in MCMC resampling

    Update the syn dataframe in place
    """
    assert m > 0
    count_con = 0

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
    db = None

    # create the df_preds generator
    if concat_attr is None:
        df_preds_gen = PredMCMC(df_preds, attr, len(df))
    else:
        df_preds_gen = PredMCMC(df_preds, concat_attr, len(df))

    # for a fixed number of resamplings
    for _ in tqdm(range(m)):

        tid = random.randint(0, len(df) - 1)
        pred_tid, pred_vals_all, pred_probas_all = df_preds_gen.get_preds(tid)
        assert tid + len(df) == pred_tid

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
                    samples = np.random.normal(pred_num, (i + 1) * std, n_val_limit).tolist()
                    pred_vals += samples

                gaussian = scipy.stats.norm(pred_num, (i + 1) * std)
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

                pred_vals = pred_vals[i * n_val_limit: (i + 1) * n_val_limit]
                pred_probas = pred_probas[i * n_val_limit: (i + 1) * n_val_limit]

                if len(pred_vals) < n_val_limit:
                    break_flag = True

                nprobas = [p / sum(pred_probas) for p in pred_probas]

            if path_weight is not None and len(relevant_dc_idx) > 0 and len(pred_vals) > 1:
                if db is None:
                    db = _prep_db_mcmc(syn, dcs, relevant_dc_idx)
                # get the number of violations if setting each value from the predication.
                # set true_count=False for faster. Return a marix of size len(pred_vals) X len(dcs)
                n_vios = get_n_vio_mcmc(syn, attr, tid, pred_vals, dcs, relevant_dc_idx, all_num_attrs, db)

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
                    logging.info(f'none of the predication is clean. tid={tid}, try {i}')
                    nprobas = [p / sum(pred_probas) for p in pred_probas]
                    pred_vals = pred_vals[:len(nprobas)]
            else:
                break_flag = True

        assert len(pred_vals) > 0
        # if attr participates in FDs, then choose the most likely value
        if attr_in_fd:
            index_max = np.argmax(nprobas)
            val = pred_vals[index_max]
        else:
            val = np.random.choice(pred_vals, p=nprobas)

        # val is the new value, assign it to syn
        old_val = syn.iloc[tid][attr]
        if old_val == val:
            count_con += 1
        else:
            syn.at[tid, attr] = val

    if db is not None:
        db.engine.dispose()

    logging.info(f'Done MCMC resampling. m= {m}, count_same= {count_con}')
