"""Utility functions for data synthesis.

"""


import logging
from string import Template
import random

import numpy as np
import pandas as pd

from holoclean.dataset import DBengine
from holoclean.dcparser import Parser
from holoclean import arguments
from synthesizer.metadata import _get_num_attrs_quant


tbl_name = 'syn'


def parse_dc(path_constraint, attrs):
    """
    Parse dc from path

    :param path_constraint path to the dcs
    :param attrs a list of all attributes

    Return a list of dcs
    """
    parser = Parser(None, None)
    parser.load_denial_constraints(path_constraint, check=False, attrs=attrs)
    dcs = parser.get_dcs()

    fd_pred_list = ['=', '<>']
    # post process to check whether each dc is fd or not. If it is, we can optimize the evaluation
    for dc in dcs:
        pred_list = [pred.operation for pred in dc.predicates]
        if set(pred_list) == set(fd_pred_list):
            dc.isFD = True
        else:
            dc.isFD = False

    return dcs


def _get_num_attrs(df, dcs):
    """
    Return a list of numeric attributes, and format categorical attribtues in the dataframe
    """
    all_num_attrs = list(df.select_dtypes(include=np.number).columns)
    attrs_in_fd = set()
    attrs_in_dc = set()
    for dc in dcs:
        if dc.isFD:
            target_set = attrs_in_fd
        else:
            target_set = attrs_in_dc
        for attr in dc.components:
            target_set.add(attr)

    num_attrs = []
    for attr in all_num_attrs:
        cast_to_str = False
        if attr in attrs_in_fd and attr not in attrs_in_dc:
            cast_to_str = True
        elif df[attr].dtype == np.int64 and (max(df[attr]) - min(df[attr]) < 200):
            cast_to_str = True

        if cast_to_str:
            df[attr] = df[attr].astype(str)
        else:
            num_attrs.append(attr)

    return num_attrs


def get_relevant_dcs(attr, sequence, dcs, exclude_fd):
    """
    Check if attr is in any dc, and that dc has all other attributes in sequence[:attr_idx]

    Return a list of dc_idx
    """
    relevant_dc_idx = []
    attr_idx = sequence.index(attr)
    for dc_idx, dc in enumerate(dcs):
        if attr in dc.components and set(dc.components).issubset(set(sequence[:attr_idx + 1])):
            if not dc.isFD or not exclude_fd:
                relevant_dc_idx.append(dc_idx)

    return relevant_dc_idx


def find_sequence(data_name, dcs, df, rand_sequence):
    attrs = list(df.columns)
    hitters = set()
    for dc_idx, dc in enumerate(dcs):
        for attr in dc.components:
            hitters.add(attr)

    if rand_sequence:
        sequence = random.sample(attrs, len(attrs))
    else:
        if 'tax' in data_name:
            sequence = ['AreaCode', 'State', 'Zip', 'City', 'HasChild', 'MaritalStatus', 'Salary', 'Rate',
                        'SingleExemp', 'ChildExemp', 'Gender', 'MarriedExemp']

            if 'part' in data_name:
                sequence = sequence[:2]

        elif 'adult' in data_name:
            if 'concat' in data_name:
                sequence = ['education', 'education-num', 'sex_income', 'race', 'relationship', 'marital-status',
                            'workclass', 'occupation', 'native-country', 'age', 'hours-per-week', 'fnlwgt',
                            'capital-gain', 'capital-loss']
            else:
                sequence = ['education', 'education-num', 'sex', 'income', 'race', 'relationship', 'marital-status',
                            'workclass', 'occupation', 'native-country', 'age', 'hours-per-week', 'fnlwgt',
                            'capital-gain', 'capital-loss']

        elif 'br2000' in data_name:
            if 'concat' in data_name:
                sequence = ['combo', 'a9', 'a12', 'a10', 'a2', 'a4', 'a3', 'a5', 'a13', 'a11']
            else:
                sequence = ['a1', 'a6', 'a7', 'a8', 'a9', 'a12', 'a14', 'a10', 'a2', 'a4', 'a3', 'a5', 'a13', 'a11']

        elif 'tpch' in data_name:
            sequence = ['c_custkey', 'c_mktsegment', 'c_nationkey', 'n_name', 'n_regionkey', 'o_orderstatus',
                        'o_orderpriority']
            # all
            # sequence = ['c_custkey', 'c_mktsegment', 'c_nationkey', 'n_name', 'n_regionkey', 'o_orderstatus',
            #             'o_orderpriority', 'c_acctbal', 'o_totalprice']

        else:
            return find_sequence_alg(data_name, dcs, df, False)

    return sequence, hitters


def find_sequence_alg(data_name, dcs, df, rand_sequence):
    """
    Default sequencing algorithm without any optimization based on domain sizes

    Find an order of schema based on a list of FDs
    :param data_name the name of this dataset
    :param dcs a list of denial constraints
    :param df dataframe
    :param rand_sequence boolean value for choosing sequnce in random, or based on constraints

    Return two values:
        - an ordered list of sequence
        - a set of attributes that participates at least one constraint
    """
    attrs = list(df.columns)
    hitters = set()
    sequence = []
    for dc_idx, dc in enumerate(dcs):
        for attr in dc.components:
            hitters.add(attr)

    if rand_sequence:
        sequence = random.sample(attrs, len(attrs))
    else:
        # get the domain size for each attributes
        attr_ds = {}
        num_attrs_quant = _get_num_attrs_quant(data_name)
        for attr in df.columns:
            if attr in num_attrs_quant:
                attr_ds[attr] = num_attrs_quant[attr]
            else:
                attr_ds[attr] = df[attr].nunique()

        fds = []
        min_domain_size = []
        for dc_idx, dc in enumerate(dcs):
            if dc.isFD:
                fds.append(dc)
                min_ds = float('inf')
                for pred_idx, pred in enumerate(dc.predicates):
                    if pred.operation == '=':
                        min_ds = min(min_ds, attr_ds[dc.components[pred_idx]])
                min_domain_size.append(min_ds)

        sorted_fds = [x for _, x in sorted(zip(min_domain_size, fds), key=lambda pair: pair[0])]
        for dc in sorted_fds:
            lhs = []
            rhs = []
            for pred_idx, pred in enumerate(dc.predicates):
                if pred.operation == '=':
                    lhs.append(dc.components[pred_idx])
                elif pred.operation == '<>':
                    rhs.append(dc.components[pred_idx])
            # sort lhs by domain size
            lds = [attr_ds[a] for a in lhs]
            sorted_lhs = [x for _, x in sorted(zip(lds, lhs))]
            for a in sorted_lhs:
                if a not in sequence:
                    sequence.append(a)
            rds = [attr_ds[a] for a in rhs]
            sorted_rhs = [x for _, x in sorted(zip(rds, rhs))]
            for a in sorted_rhs:
                if a not in sequence:
                    sequence.append(a)

        # append other attributes by domain size
        others = []
        others_ds = []
        for a in df.columns:
            if a in sequence:
                continue
            others.append(a)
            others_ds.append(attr_ds[a])
        sorted_others = [x for _, x in sorted(zip(others_ds, others))]
        sequence.extend(sorted_others)

    return sequence, hitters


def _check_vio_row(df, tid, dcs, relevant_dc_idx=None):
    """
    The function to count the number of violation introduced by row (indexed by tid) w.r.t the rest dataset

    :param df dataframe holding the dataset
    :param tid tid for the row
    :param dcs a list of denial constraints
    :param relevant_dc_idx a list of relevant dc idxes that only need to consider

    Return a list of integers, each value corresponds to the number of violations to one dc.
    """
    n_vios = [0] * len(dcs)
    row = df.iloc[tid]

    for dc_idx, dc in enumerate(dcs):
        if relevant_dc_idx is not None:
            if dc_idx not in relevant_dc_idx:
                continue
        # compute n_vio for tid on dc
        n_vio = 0
        vals = [row[attr] for attr in dc.components]
        # skip if this contains empty value
        if np.nan in vals:
            continue
        operations = [pred.operation for pred in dc.predicates]
        if dc.isFD:
            df_filter = df.drop([tid])
            for idx in range(len(operations)):
                operation = operations[idx]
                if operation == '=':
                    df_filter = df_filter[df_filter[dc.components[idx]] == vals[idx]]
                elif operation == '<>':
                    df_filter = df_filter[df_filter[dc.components[idx]] != vals[idx]]
            n_vio = df_filter.shape[0]
        else:
            pass

        n_vios[dc_idx] = n_vio

    return n_vios


def _check_vio_row(df, tid, dcs, ds, data_name, relevant_dc_idx=None):
    """
    The function to count the number of violation introduced by row (indexed by tid) w.r.t the rest dataset

    :param df the dataframe
    :param tid tid for the row
    :param dcs a list of denial constraints
    :param ds an instance of HoloClean dataset
    :param data_name string for the table name in postgres
    :param relevant_dc_idx a list of relevant dc idxes that only need to consider

    Return a list of integers, each value corresponds to the number of violations to one dc.
    """
    n_vios = [0] * len(dcs)
    row = df.iloc[tid]

    sql1_template = f"select count(*) from {data_name} t1 where t1._tid_={tid} and "
    sql2_template = f"select count(*) from {data_name} t1, {data_name} t2 where t1._tid_={tid} and t2._tid_!={tid} and "
    queries = []
    for dc_idx, dc in enumerate(dcs):
        if relevant_dc_idx is not None:
            if dc_idx not in relevant_dc_idx:
                continue
        # compute n_vio for tid on this dc
        vals = [row[attr] for attr in dc.components]
        # skip if this contains empty value
        if np.nan in vals:
            continue

        if len(dc.tuple_names) == 2:
            sql = sql2_template + dc.cnf_form
        elif len(dc.tuple_names) == 1:
            sql = sql1_template + dc.cnf_form
        else:
            sql = None

        queries.append(sql)

    re = ds.engine.execute_queries(queries)

    if relevant_dc_idx is None:
        if len(dcs) != len(queries):
            print(f"Error: {dc_idx}, {len(dcs)}, {len(queries)}")
        assert len(dcs) == len(queries)
        for idx in range(len(dcs)):
            n_vios[idx] = re[idx][0][0]
    else:
        re_idx = 0
        for dc_idx, dc in enumerate(dcs):
            if dc_idx in relevant_dc_idx:
                n_vios[dc_idx] = re[re_idx][0][0]
                re_idx += 1
            else:
                n_vios[dc_idx] = 0
        assert re_idx == len(re)

    return n_vios


def _get_env():
    """
    Internal function to get HoloClean database config for accessing postgres
    """
    arg_defaults = {}
    for arg, opts in arguments:
        if 'directory' in arg[0]:
            arg_defaults['directory'] = opts['default']
        else:
            arg_defaults[opts['dest']] = opts['default']

    return arg_defaults


def _get_weights(path_weight):
    """
    Read weights from file
    """
    weights = []
    with open(path_weight, 'r') as file:
        found = False
        for line in file:
            if not found and not line.startswith('++'):
                continue
            elif line.startswith('++'):
                found = True
            else:
                weights.append(float(line))

    return weights


def _update_cnf(cnf, row, all_num_attrs):
    """
    Replace the dc cnf_form with values. E.g.,
        cnf: t1."Zip"=t2."Zip" AND t1."City"<>t2."City"
        replace to cnf: t1."Zip"="98052" AND t1."City"<>"Redmond"
    """
    new_cnf = str(cnf)
    for (k, v) in row.iteritems():
        sql_string = f't2."{k}"'
        if k in all_num_attrs:
            new_sql_string = str(v)
        else:
            new_sql_string = f"'{v}'"
        new_cnf = new_cnf.replace(sql_string, new_sql_string)

    return new_cnf


sql2_template_count = Template('select count(*) from "$tbl_name" t1 where t1."tid" < $tid and $cnf')
sql2_template_flag = Template('select case '
                              'when exists ( select * from "$tbl_name" t1 where t1."tid"< $tid and $cnf) '
                              'then 1 else 0 end')


def get_n_vio(syn, syn_values, attr, vals, dcs, relevant_dc_idx, all_num_attrs, db, true_count):
    """
    Compute the number of violations if assigning attr with val

    Return 2D list of size len(vals) X len(dcs). Each row corresponds the violations of a potential value.
    """
    tid = len(syn_values)
    row = syn.iloc[tid].copy()

    if len(syn_values) > 1:
        # insert the chosen value from last run into the table
        pre_val = syn_values[-1]
        pre_tid = tid - 1
        if attr in all_num_attrs:
            sql_update = f'update {tbl_name} set "{attr}"={pre_val} where tid={pre_tid}'
        else:
            sql_update = f'update {tbl_name} set "{attr}"=\'{pre_val}\' where tid={pre_tid}'

        db.execute_query_no_return(sql_update)

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
                if true_count:
                    sql = sql2_template_count.substitute(tbl_name=tbl_name, tid=tid, cnf=replaced_cnf)
                else:
                    sql = sql2_template_flag.substitute(tbl_name=tbl_name, tid=tid, cnf=replaced_cnf)
            elif len(dc.tuple_names) == 1:
                sql = None
            else:
                sql = None

            sql_idx_to_matrix_idx[len(queries)] = (val_idx, dc_idx)
            queries.append(sql)

            re = db.execute_query(sql)
            n_vios[val_idx, dc_idx] = re[0][0]

    return n_vios


def _prep_db(syn, syn_values, attr, atype, dcs, relevant_dc_idx):
    """
    Create the synthetic table, which will be used to get the number of violations (function get_n_vio)
    """
    df = syn.copy()
    df[attr] = [syn_values[0]] * len(syn)
    df[attr] = df[attr].astype(atype)

    # create a db engine
    env = _get_env()
    db = DBengine(env['db_user'], env['db_pwd'], env['db_name'], env['db_host'],
                  pool_size=env['threads'], timeout=env['timeout'])

    df.to_sql(tbl_name, con=db.engine, if_exists='replace', index=True, index_label='tid')

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


def _get_sampling_paras(AR=False):
    if AR:
        return 1, 300
    else:
        return 20, 3


def revert_tax(path, a1, a2, a3, unselected_a2_4_a1, unselected_a3_4_a1):
    df = pd.read_csv(path)

    # write to backup file
    df.to_csv(f'{path}.backup', index=False)

    new_a2s = []
    new_a3s = []
    for idx, row in df.iterrows():
        v1 = row[a1]
        v2 = row[a2]
        v3 = row[a3]
        rplc_str_a2 = f'oa2'
        rplc_str_a3 = f'oa3'

        if rplc_str_a2 in v2:
            new_a2s.append(random.choice(tuple(unselected_a2_4_a1[v1])))
        else:
            new_a2s.append(v2)
        if rplc_str_a3 in v3:
            new_a3s.append(random.choice(tuple(unselected_a3_4_a1[v1])))
        else:
            new_a3s.append(v3)

    df[a2] = new_a2s
    df[a3] = new_a3s

    df.to_csv(path, index=False)


class Pred:
    def __init__(self, df_preds, attr, start_tid):
        self.df_preds = df_preds
        self.attr = attr
        self.start_tid = start_tid
        self.yield_count = 0

    def __iter__(self):
        pred_vals = []
        pred_probas = []
        for pred in self.df_preds:
            if pred['tid'] < self.start_tid or pred['attribute'] != self.attr:
                continue

            if pred['tid'] == self.start_tid + self.yield_count:
                pred_vals.append(pred['inferred_val'])
                pred_probas.append(pred['proba'])
            else:
                yield self.start_tid + self.yield_count, pred_vals, pred_probas

                pred_vals.clear()
                pred_probas.clear()

                self.yield_count += 1
                pred_vals.append(pred['inferred_val'])
                pred_probas.append(pred['proba'])

        yield self.start_tid + self.yield_count, pred_vals, pred_probas
