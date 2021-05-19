# Kamino: Constraint-Aware Differentially Private Data Synthesis

Kamino is a data synthesis system to ensure differential privacy and to preserve the structure and correlations present in the original dataset. Kamino takes as input of a database instance, along with its schema (including integrity constraints), and produces a synthetic database instance with differential privacy and structure preservation guarantees.

Full research paper is available on arXiv: [Kamino: Constraint-Aware Differentially Private Data Synthesis](https://arxiv.org/abs/2012.15713).

## Installation

Kamino was built using Python and PostgreSQL as the backend data management. This codebase uses Conda and Docker to manage the runtime environment.



### 1. Install environment

Please follow the instructions to get [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) and [Docker](https://docs.docker.com/get-docker/).

### 2. Setup Kamino

#### 2.1 Clone the code
As the first step, clone this repository as well as submodules:
```bash
$ git clone --recurse-submodules https://github.com/cgebest/kamino.git 
$ cd kamino
```

#### 2.2 Install Python dependencies
Create a Python environment, install all dependencies and set the environment variable:
```bash
$ conda env create -f=environment.yml
$ conda activate kamino
$ export PYTHONPATH=$(pwd)
```

#### 2.3 Create the backend database
Run the script to create a PostgresSQL image using Docker, on port 5432:
```bash
$ ./create_db.sh
```

#### 2.4 Test your setup
Run the sanity check: 
```bash
$ python ./synthesizer/test_env.py
```
The `test_env.py` synthesizes a toy dataset and evaluates on it. If your environment has been properly set, the sanity check will print success and end normally.  

## Run Kamino

We provide 4 datasets that have been tested in our experiments. They are located in the directory of `Kamino/testdata/`. Within each dataset folder, there are 3 files that are input to Kamino:
- `*.csv` The dataset in CSV format with header.
- `*.ic` The set of denial constraint (DC).
- `*.w` The weight of each DC.

### 1. Test Kamino on the provided datasets

#### 1.1 Run with the default settings
We provide one-click scripts for each of the four datasets: `adult`, `br2000`, `tax` and `tpch`. The parameters are the default settings described in our paper. In particular, the differential privacy budget is set to be (epsilon=1, delta=1e-6).

To generate a synthetic dataset, simply run one of the `syn_[dataset].py` files:
```bash
$ python ./synthesizer/syn_[dataset].py
```
The script outputs the synthetic dataset, and also automatically evaluates by conducting three tasks: DC violations, model training, and 1-/2-way marginal distances. The output will be saved in the directory `./testdata/[dataset]/[data]-[time]_[hostname]/`, where `*.syn` is the generated synthetic data, and `*.log` is the log file with evaluations.

Depending on the hardware, it can take a few hours to complete.

#### 1.2 Parameter options 

The main place to config the parameters is the `paras` dictionary. Here is an example with explanations:

```python
paras = {
    'reuse_embedding': True,  # True to reuse the tupple embedding
    'dp': True,  # True to run in differtial privacy (DP)
    'n_row': n_row,  # number of rows in the true dataset
    'n_col': n_col,  # number of columns in the true dataset
    'epsilon1': .1,  # DP parameter for generating first attribtue
    'l2_norm_clip': 1.0, # DP parameter for clipping the gradient norm
    'noise_multiplier': 1.1,  # DP parameter for gaussian coefficient
    'minibatch_size': 23,  # batch size to sample for each iteration
    'microbatch_size': 1,  # micro batch size
    'delta': 1e-6,  # delta in DP budget
    'learning_rate': 1e-4,  # learning rate
    'iterations': 1600,  # number of iterations
    'AR': False,  # True to enable accept-reject sampling
    'MCMC': 0,  # number of mcmc re-sampling cells in percentage of n_row
}
```

To disable task evaluation, simply comment out the `evaluate_data()` function call.

### 2. Run Kamino on your own dataset

#### 2.1 Prepare the inputs 
For each dataset, three input files are required. You can take a look at the provided inputs and prepare the following:
- `[dataset].csv` The dataset in CSV format with header. Rows with nan will be dropped by Kamino.
- `[dataset].ic` Please follow the grammar defined by the DC parser `Kamino/dcparser/constraint.py`.
- `[dataset].w` The weight of each DC. Either outputted by `synthesizer/ICweight.py` (see `synthesizer/syn_br2000.py` for an example) or manual specified (for hard DCs, check `testdata/adult/adult.w`).

In addition, training model requires quantization information for numerical attributes (in `metadata.py`).

#### 2.2 Config the parameters

Duplicate one of the `syn_[dataset].py` files and fill your inputs. If you see any issues when running your dataset, run in debug mode and pinpoint the places.  


Depending on your purpose and dataset, you may set the parameters differently. If `dp=True`,  the total privacy cost is calculated by the `analysis.epsilon()` function.

## Acknowledgement and License
Kamino reuses [HoloClean](https://github.com/HoloClean/holoclean) and [PyVacy](https://github.com/ChrisWaites/pyvacy) as submodules. Major modifications to their original repos include:
- Add private training of the HoloClean models.
- Add cache and batch prediction dump in HoloClean.
- Add one-shot privacy composition in PyVacy.

[Apache-2.0 License](LICENSE).

## Questions?

This repository was implemented for research PoC only. 

For any questions or comments, please reach out to Chang Ge (chang.ge AT uwaterloo.ca).