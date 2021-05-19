"""Functions to pre- and post-process data.

"""
import os
import random
import re

import pandas as pd
import numpy as np


def preproc_tax():
    path_data = f'./testdata/tax/tax30k.csv'
    tmp_path = './testdata/tax/tax.poc'
    df = pd.read_csv(path_data)

    sequence = ['AreaCode', 'State', 'Zip', 'City', 'HasChild', 'MaritalStatus', 'Salary', 'Rate', 'SingleExemp',
                'ChildExemp', 'Gender', 'MarriedExemp']

    df.dropna(axis=0, inplace=True)

    states = list(df['State'].unique())
    zips = list(df['Zip'].unique())

    selected_zip4state = {}
    replaced_zip4state = {}
    # threshold for spare vector
    T = 3

    for state in states:
        value_counts = df[df['State'] == state]['Zip'].value_counts()
        active_zips = list(value_counts.index)
        selected_zips = []
        replaced_zips = []
        for zip in zips:
            true_count = 0 if zip not in active_zips else value_counts[zip]

            if true_count >= T:
                selected_zips.append(zip)
            else:
                replaced_zips.append(zip)

        selected_zip4state[state] = selected_zips
        replaced_zip4state[state] = replaced_zips

    # replace zip and city in df
    new_zips = []
    new_cities = []
    for idx, row in df.iterrows():
        state = row['State']
        zip = row['Zip']
        city = row['City']
        rplc_str_zip = f'{state}_ozip'
        rplc_str_city = f'{state}_ocity'
        if zip not in selected_zip4state[state]:
            new_zips.append(rplc_str_zip)
            new_cities.append(rplc_str_city)
        else:
            new_zips.append(zip)
            new_cities.append(city)
    df['Zip'] = new_zips
    df['City'] = new_cities

    df.to_csv(tmp_path, index=False)


def postproc_tax(path_true, path_syn):

    df_true = pd.read_csv(path_true)
    df_syn = pd.read_csv(path_syn)

    df_true.dropna(axis=0, inplace=True)

    zips = list(df_true['Zip'].unique())
    cities = list(df_true['City'].unique())

    col_city = []
    col_zip = []
    for idx, row in df_syn.iterrows():
        zip = row['Zip']
        city = row['City']

        zip_v = zip
        if '_ozip' in zip:
            zip_v = random.choice(zips)
        col_zip.append(zip_v)

        city_v = city
        if '_ocity' in city:
            city_v = random.choice(cities)
        col_city.append(city_v)

    df_syn['Zip'] = col_zip
    df_syn['City'] = col_city

    base = os.path.basename(path_syn)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path_syn)
    out_path = f'{dir_path}/{data_name}_post.syn'
    df_syn.to_csv(out_path, index=False)


def preproc_br2000(epsilon=None):
    path_data = f'./testdata/br2000/br2000.csv'
    tmp_path = './testdata/br2000/br2000.poc'

    df = pd.read_csv(path_data)
    binary_attrs = {'a1': [], 'a6': [], 'a7': [], 'a8': [], 'a9': [], 'a12': [], 'a14': []}
    df_combo = df[binary_attrs.keys()].agg('_'.join, axis=1)

    battrs = ['zero', 'one']
    domain = []
    counts = []
    for v1 in battrs:
        for v6 in battrs:
            for v7 in battrs:
                for v8 in battrs:
                    for v9 in battrs:
                        for v12 in battrs:
                            for v14 in battrs:
                                v = f'{v1}_{v6}_{v7}_{v8}_{v9}_{v12}_{v14}'
                                true_count = sum(df_combo == v)
                                domain.append(v)
                                counts.append(true_count)

    if epsilon is not None:
        noise = np.random.laplace(0, 1.0 / epsilon, len(domain))
        noisy_freq = []
        for i in range(len(domain)):
            count = counts[i] + noise[i]
            count = max(0, count)
            noisy_freq.append(count / df.shape[0])
    else:
        noisy_freq = counts

    noisy_freq = [f / sum(noisy_freq) for f in noisy_freq]

    all_lists = list(binary_attrs.values())
    for _ in range(df.shape[0]):
        sampled_combo = np.random.choice(domain, p=noisy_freq)
        splits = sampled_combo.split('_')
        assert len(splits) == len(all_lists)
        for i in range(len(all_lists)):
            all_lists[i].append(splits[i])

    syn = pd.DataFrame(binary_attrs)

    syn.to_csv(tmp_path, index=False)
    return tmp_path


def preproc_br2000_separate(path):
    """
    Group bianry attributes as one attribute. Only execute once.
    """
    df = pd.read_csv(path)
    target_attrs = ['a1', 'a6', 'a7', 'a8', 'a14']
    df['combo'] = df[target_attrs].agg('_'.join, axis=1)
    df.drop(columns=target_attrs, inplace=True)

    # iid_attrs = ['a5', 'a13']
    # df.drop(columns=iid_attrs, inplace=True)

    base = os.path.basename(path)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path)

    out_path = f'{dir_path}/{data_name}_concat.csv'
    df.to_csv(out_path, index=False)

    return out_path


def postproc_br2000(path):
    """"
    Restore combo into individual attributes
    """
    df = pd.read_csv(path)
    df[['a1', 'a6', 'a7', 'a8', 'a14']] = df.combo.str.split("_", expand=True)
    df.drop(columns=['combo'], inplace=True)

    out_path = re.sub("\_concat", '', path)
    df.to_csv(out_path, index=False)

    return out_path


def preproc_adult(path):
    """
    Group sex and income as one attribute. Only execute once.
    """
    df = pd.read_csv(path)
    target_attrs = ['sex', 'income']
    df['sex_income'] = df[target_attrs].agg('_'.join, axis=1)
    df.drop(columns=target_attrs, inplace=True)

    base = os.path.basename(path)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path)

    out_path = f'{dir_path}/{data_name}_concat.csv'
    df.to_csv(out_path, index=False)

    return out_path


def postproc_adult(path):
    """"
    Restore the adult whose sex and income are concat
    """
    df = pd.read_csv(path)
    df[['sex', 'income']] = df.sex_income.str.split("_", expand=True)
    df.drop(columns=['sex_income'], inplace=True)

    out_path = re.sub("\_concat", '', path)
    df.to_csv(out_path, index=False)

    return out_path


if __name__ == '__main__':

    # preproc_tax()
    preproc_br2000()
