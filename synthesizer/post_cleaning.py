import os
import re

import pandas as pd

import holoclean
from holoclean.detect import *
from holoclean.repair.featurize import ConstraintFeaturizer


def repair(path_data, path_ic):
    """
    The repair dataset will be stored at the same directory as input
    """
    hc = holoclean.HoloClean(domain_thresh_1=0.0, domain_thresh_2=0.0, max_domain=5000,
                             cor_strength=0, weight_decay=0., learning_rate=0.001, threads=1, batch_size=1,
                             verbose=False, timeout=36000, infer_mode='dk').session

    base = os.path.basename(path_data)
    data_name = os.path.splitext(base)[0]
    dir_path = os.path.dirname(path_data)
    output_path = f'{dir_path}/{data_name}_repaired.csv'

    hc.load_data(data_name, path_data)
    hc.load_dcs(path_ic)
    hc.ds.set_constraints(hc.get_dcs())

    detectors = [ViolationDetector()]
    hc.detect_errors(detectors)

    hc.generate_domain()
    hc.run_estimator()
    featurizers = [ConstraintFeaturizer()]

    hc.repair_errors(featurizers)

    df_repaired = hc.ds.repaired_data.df.copy()
    df_repaired.drop(columns=['_tid_'], inplace=True)
    df_repaired.to_csv(output_path, index=False)

    print(f'Done repairing\n{output_path}')
    return output_path


def _restore_full_tax(path_true, path_syn):
    true_size = pd.read_csv(path_true).shape[0]
    df = pd.read_csv(path_syn)

    frac = 1. * true_size / df.shape[0]

    df_full = df.sample(frac=frac, replace=True, random_state=0)

    new_path_syn = re.sub('\_5k', '', path_syn)
    df_full.to_csv(new_path_syn, index=False)


if __name__ == '__main__':

    # path_tax_true = './testdata/tax/tax30k.csv'

    path_syns = [

    ]

    for path_data in path_syns:
        path_ic = './testdata/adult/adult.ic'
        # path_ic = './testdata/tax/tax.ic'
        # path_ic = './testdata/br2000/br2000.ic'

        output_path = repair(path_data, path_ic)

        # _restore_full_tax(path_tax_true, output_path)
