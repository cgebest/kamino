import logging
import time

import pandas as pd
import numpy as np

from pyvacy.pyvacy import analysis
from synthesizer.kamino import synthesize_continue, syn_data
from synthesizer.util import _analyze_privacy, evaluate_data, copy_log


def _syn_tax_iid_num(path_syn, std1):
    """
    Internal use for generating the last a few numerical attributes
    """
    df_syn = pd.read_csv(path_syn)
    path_data = f'./testdata/tax/tax30k.csv'
    df = pd.read_csv(path_data)
    df.dropna(axis=0, inplace=True)

    # 'Salary', 'Rate',
    # 'SingleExemp', 'ChildExemp', 'Gender', 'MarriedExemp'
    num_attrs = ['Salary', 'Rate', 'SingleExemp', 'ChildExemp', 'MarriedExemp']

    n_bins = 50
    bin_widths = {}
    for attr in num_attrs:
        df_syn[attr] = pd.cut(df[attr], n_bins, include_lowest=True, labels=np.array(range(n_bins)))
        bin_widths[attr] = (max(df[attr]) - min(df[attr])) / n_bins

    num_attrs_iid = ['Salary', 'MarriedExemp']
    for attr in num_attrs_iid:
        print(attr)
        domain = range(n_bins)
        temp_df_attrs = df_syn[attr].value_counts()
        probas_values = []
        for value in domain:
            count = temp_df_attrs[value]
            noise = np.random.normal(0, std1)
            # noise = 0
            noisy_count = max(0, count + noise)
            probas_values.append(noisy_count)
        probas_values = np.array(probas_values) / np.sum(probas_values)

        syn_values = []
        while len(syn_values) < df_syn.shape[0]:
            bin_num = np.random.choice(domain, p=probas_values)
            left = bin_num * bin_widths[attr]
            syn_value = np.random.uniform(left, left + bin_widths[attr])
            syn_values.append(syn_value)

        df_syn[attr] = syn_values
        df_syn[attr] = df_syn[attr].astype(df[attr].dtype)

    # Rate, t1&t2&EQ(t1.State,t2.State)&GT(t1.Salary,t2.Salary)&LT(t1.Rate,t2.Rate)
    # ChildExemp, t1&t2&EQ(t1.State,t2.State)&EQ(t1.HasChild,t2.HasChild)&IQ(t1.ChildExemp,t2.ChildExemp)
    # SingleExemp, t1&t2&EQ(t1.State,t2.State)&EQ(t1.MaritalStatus,t2.MaritalStatus)&IQ(t1.SingleExemp,t2.SingleExemp)
    num_attrs_dcs = ['Rate', 'SingleExemp', 'ChildExemp']
    syn_states = df_syn['State'].tolist()
    syn_salaries = df_syn['Salary'].tolist()
    syn_haschilds = df_syn['HasChild'].tolist()
    syn_maritalstatus = df_syn['MaritalStatus'].tolist()

    for attr in num_attrs_dcs:
        print(attr)
        probas_values = []
        domain = range(n_bins)
        temp_df_attrs = df_syn[attr].value_counts()
        for value in domain:
            count = temp_df_attrs[value]
            noise = np.random.normal(0, std1)
            # noise = 0
            noisy_count = max(0, count + noise)
            probas_values.append(noisy_count)
        probas_values = np.array(probas_values) / np.sum(probas_values)

        syn_values = []
        if attr == 'Rate':
            while len(syn_values) < df_syn.shape[0]:
                idx = len(syn_values)
                crnt_state = syn_states[idx]
                crnt_salary = syn_salaries[idx]

                bin_num = np.random.choice(domain, p=probas_values)
                left = bin_num * bin_widths[attr]
                crnt_rate = np.random.uniform(left, left + bin_widths[attr])

                # first loop, check if there is a tuple with same state and salary
                equal_flag = False
                for pre_idx in range(len(syn_values)):
                    pre_state = syn_states[pre_idx]
                    pre_salary = syn_salaries[pre_idx]
                    pre_rate = syn_values[pre_idx]

                    if pre_state == crnt_state and pre_salary == crnt_salary:
                        crnt_rate = pre_rate
                        equal_flag = True
                        break

                if not equal_flag:
                    # second loop, find max rate, for salary less than current salary
                    max_pre_salary = None
                    max_pre_rate = None
                    for pre_idx in range(len(syn_values)):
                        pre_state = syn_states[pre_idx]
                        pre_salary = syn_salaries[pre_idx]
                        if pre_state != crnt_state:
                            continue
                        if pre_salary > crnt_salary:
                            continue
                        pre_rate = syn_values[pre_idx]
                        if max_pre_salary is None or max_pre_salary < pre_salary:
                            max_pre_salary = pre_salary
                            max_pre_rate = pre_rate

                    # third loop, find min rate, for salary higher than current salary
                    min_pre_salary = None
                    min_pre_rate = None
                    for pre_idx in range(len(syn_values)):
                        pre_state = syn_states[pre_idx]
                        pre_salary = syn_salaries[pre_idx]
                        if pre_state != crnt_state:
                            continue
                        if pre_salary < crnt_salary:
                            continue
                        pre_rate = syn_values[pre_idx]
                        if min_pre_salary is None or min_pre_salary > pre_salary:
                            min_pre_salary = pre_salary
                            min_pre_rate = pre_rate

                    if max_pre_salary is None:
                        max_pre_rate = min(df['Rate'])
                    if min_pre_salary is None:
                        min_pre_rate = max(df['Rate'])

                    if max_pre_rate <= crnt_rate <= min_pre_rate:
                        pass
                    else:
                        crnt_rate = np.random.uniform(max_pre_rate, min_pre_rate)

                syn_values.append(crnt_rate)

        elif attr == 'SingleExemp':
            # t1&t2&EQ(t1.State,t2.State)&EQ(t1.MaritalStatus,t2.MaritalStatus)&IQ(t1.SingleExemp,t2.SingleExemp)
            while len(syn_values) < df_syn.shape[0]:
                idx = len(syn_values)
                crnt_state = syn_states[idx]
                crnt_maritalstatus = syn_maritalstatus[idx]

                bin_num = np.random.choice(domain, p=probas_values)
                left = bin_num * bin_widths[attr]
                crnt_value = np.random.uniform(left, left + bin_widths[attr])

                for pre_idx in range(len(syn_values)):
                    pre_state = syn_states[pre_idx]
                    pre_maritalstatus = syn_maritalstatus[pre_idx]
                    pre_value = syn_values[pre_idx]

                    if pre_state == crnt_state and pre_maritalstatus == crnt_maritalstatus:
                        crnt_value = pre_value
                        break
                syn_values.append(crnt_value)

        elif attr == 'ChildExemp':
            # ChildExemp, t1&t2&EQ(t1.State,t2.State)&EQ(t1.HasChild,t2.HasChild)&IQ(t1.ChildExemp,t2.ChildExemp)
            while len(syn_values) < df_syn.shape[0]:
                idx = len(syn_values)
                crnt_state = syn_states[idx]
                crnt_haschild = syn_haschilds[idx]

                bin_num = np.random.choice(domain, p=probas_values)
                left = bin_num * bin_widths[attr]
                crnt_value = np.random.uniform(left, left + bin_widths[attr])

                for pre_idx in range(len(syn_values)):
                    pre_state = syn_states[pre_idx]
                    pre_haschild = syn_haschilds[pre_idx]
                    pre_value = syn_values[pre_idx]

                    if pre_state == crnt_state and pre_haschild == crnt_haschild:
                        crnt_value = pre_value
                        break

                syn_values.append(crnt_value)

        df_syn[attr] = syn_values.copy()
        df_syn[attr] = df_syn[attr].astype(df[attr].dtype)

    df_syn.to_csv(path_syn, index=False)


def syn_tax():
    """
    The full tax with city and zip - private setup
    use iid for zip and city
    Revision note: this is actually the chosen setup
    """
    start = time.time()

    path_data = f'./testdata/tax/tax30k.csv'
    path_ic = f'./testdata/tax/tax30k.ic'

    path_data_part = f'./testdata/tax/tax30k_part.csv'
    path_ic_part = f'./testdata/tax/tax30k_part.ic'

    df = pd.read_csv(path_data)

    n_row, n_col = df.shape
    n_len = len(str(n_row)) + 1

    paras = {
        'reuse_embedding': True,  # set True to reuse the embedding
        'dp': True,  # set True to enable privacy
        'n_row': n_row,  # number of rows in the true dataset
        'n_col': n_col - 2,  # number of columns in the true dataset
        'epsilon1': .4,  # epsilon1, for iid
        'l2_norm_clip': 1.0,
        'noise_multiplier': 1.1,
        'minibatch_size': 18,  # batch size to sample for each iteration.
        'microbatch_size': 1,  # micro batch size
        'delta': float(f'1e-{n_len}'),  # depends on data size. Do not change for now
        'learning_rate': 1e-4,
        'iterations': 2000  # number of iteration. Should be large enough: iterations * minibatch_size > N
    }

    if paras['dp']:
        epsilon2 = _analyze_privacy(paras)
        paras['epsilon2'] = epsilon2

        gaussian_std = []
        sensitivity = 2
        std1 = np.sqrt(sensitivity * 2. * np.log(1.25 / paras['delta'])) / paras['epsilon1']
        for i in range(3):
            gaussian_std.append(std1)

        epsilon = analysis.epsilon(N=n_row,
                                   batch_size=paras['minibatch_size'],
                                   noise_multiplier=paras['noise_multiplier'],
                                   iterations=paras['iterations'] * (paras['n_col']-1),
                                   delta=paras['delta'], gaussian_std=gaussian_std)

        msg = f"epsilon1= {3 * paras['epsilon1']}\tepsilon2= {'{:.2f}'.format(epsilon2)}\t" \
              f"delta= {paras['delta']}\nepsilon={epsilon}"
        print(msg)

    syn_data(path_data_part, path_ic_part, paras)
    path_syn_part = paras['path_syn']

    # continue on the 'Zip', 'City'
    zips = list(df['Zip'].unique())
    cities = list(df['City'].unique())
    probas_zips = []
    probas_cities = []

    temp_df_zips = df['Zip'].value_counts()
    for crnt_zip in zips:
        count = temp_df_zips[crnt_zip]
        # noise = np.random.laplace(0, 1.0 / paras['epsilon1'])
        noise = np.random.normal(0, std1)
        noisy_count = max(0, count + noise)
        probas_zips.append(noisy_count)
    probas_zips = np.array(probas_zips) / np.sum(probas_zips)

    temp_df_cities = df['City'].value_counts()
    for city in cities:
        count = temp_df_cities[city]
        # noise = np.random.laplace(0, 1.0 / paras['epsilon1'])
        noise = np.random.normal(0, std1)
        noisy_count = max(0, count + noise)
        probas_cities.append(noisy_count)
    probas_cities = np.array(probas_cities) / np.sum(probas_cities)

    df_syn = pd.read_csv(path_syn_part)

    syn_zips = []
    syn_cities = []
    syn_states = df_syn['State'].tolist()

    while len(syn_zips) < len(syn_states):
        idx = len(syn_zips)
        crnt_state = syn_states[idx]
        crnt_zip = np.random.choice(zips, p=probas_zips)
        previous_city = None
        # check if crnt_zip violates zip->state
        if crnt_zip in syn_zips:
            previous_idx = syn_zips.index(crnt_zip)
            previous_state = syn_states[previous_idx]
            previous_city = syn_cities[previous_idx]
            if previous_state != crnt_state:
                continue

        # so far, zip->states
        if previous_city is None:
            crnt_city = np.random.choice(cities, p=probas_cities)
        else:
            crnt_city = previous_city

        syn_zips.append(crnt_zip)
        syn_cities.append(crnt_city)

    df_syn['Zip'] = syn_zips
    df_syn['City'] = syn_cities
    tmp_syn_path = '/tmp/tax_iid.csv'
    df_syn.to_csv(tmp_syn_path, index=False)

    synthesize_continue(path_data=path_data, path_syn_old=tmp_syn_path, path_constraint=path_ic,
                        paras=paras, rand_sequence=False, ic_sampling=True)

    path_syn = paras['path_syn']

    end = time.time()
    time_wall = end - start
    paras['TIME_WALL'] = time_wall
    time_sampling = time_wall - paras['TIME_TRAINING']
    logging.info(f"TIME_TRAINING= {paras['TIME_TRAINING']}")
    logging.info(f'TIME_SAMPLING= {time_sampling}')
    logging.info(f'TIME_ALL= {time_wall}')

    evaluate_data(path_data, path_syn, path_ic)

    copy_log(paras)


if __name__ == '__main__':
    """
    The entry point for using kamino to generate synthetic dataset
    """

    syn_tax()

