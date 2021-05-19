"""Functions to facilitate tests.

"""


import logging
import shutil

from pyvacy.pyvacy import analysis
from synthesizer.evaluation import validate_dc_vio, validate_accuracy, validate_kway_marginal


logging_file = "kamino.log"
logging.basicConfig(filename=logging_file, filemode='w',
                    format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


def _analyze_privacy(paras):
    epsilon2 = analysis.epsilon(N=paras['n_row'], batch_size=paras['minibatch_size'],
                                noise_multiplier=paras['noise_multiplier'], delta=paras['delta'],
                                iterations=paras['iterations'] * (paras['n_col'] - 1))

    return epsilon2


def evaluate_data(path_data, path_syn, path_ic):

    validate_dc_vio(path_ic, path_syn)

    validate_kway_marginal(1, path_data, path_syn)
    validate_kway_marginal(2, path_data, path_syn)

    validate_accuracy(path_data, path_syn)


def copy_log(paras):
    dir_syn = paras['dir_syn']
    data_name = paras['synfile_name'][:-4]
    # move log to dir_syn
    shutil.move(logging_file, f'{dir_syn}/{data_name}.log')

    # shared_path = os.path.expanduser('~/holomake_output/')
    # os.makedirs(shared_path, exist_ok=True)
    # os.system(f'cp -r {dir_syn} {shared_path}')


TPCH_ONLY = False


def get_tpch_only():
    return TPCH_ONLY


def set_tpch_only():
    global TPCH_ONLY
    TPCH_ONLY = True

