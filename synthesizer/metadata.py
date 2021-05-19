"""Functions to dock the metadata used for the test datasets.

"""


def _get_num_attrs_quant(data):
    """
    Return the quantization info for numeric attributes
    """
    if 'adult' in data:
        num_attrs_quant = {'fnlwgt': 100, 'capital-gain': 100, 'capital-loss': 100}
    elif 'tax' in data:
        num_attrs_quant = {'Salary': 50, 'Rate': 10, 'SingleExemp': 50, 'MarriedExemp': 50, 'ChildExemp': 50}
    elif 'br2000' in data:
        num_attrs_quant = {}
    elif 'tpch' in data:
        num_attrs_quant = {'c_acctbal': 100, 'o_totalprice': 100}
    else:
        print(f'No quantization info set for {data}')
        num_attrs_quant = {}

    return num_attrs_quant


def _get_targets(path):
    """
    Return target and postive labels for ML models
    """
    target_attrs = []
    pos_values = []

    if 'adult' in path:
        target_attrs = ['income', 'sex', 'marital-status', 'education',
                        'age', 'workclass', 'education-num',
                        'occupation', 'relationship', 'race',
                        'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country'
                        ]
        pos_values = [['>50k'], ['female'], ['never-married'],
                      ['some-college', 'assoc-voc', 'assoc-acdm', 'masters', 'doctorate', 'bachelors', 'prof-school'],
                      [50, 100],
                      ['state-gov', 'federal-gov', 'local-gov'],
                      [10, 17],
                      ['prof-specialty'],
                      ['unmarried'],
                      ['black'],
                      [0, 1000], [0, 1000],
                      [40, 100],
                      ['united-states']
                      ]
    elif 'tax' in path:
        target_attrs = ['Gender',
                        'AreaCode',
                        'City',
                        'State',
                        'Zip',
                        'MaritalStatus', 'HasChild',
                        'Salary',
                        'Rate',
                        'SingleExemp',
                        'MarriedExemp',
                        'ChildExemp'
        ]
        pos_values = [['f'],
                      [f'ac{code}' for code in range(300, 400)],
                      ['watkins', 'waterbury', 'waters', 'waterproof', 'watrous', 'watkinsville', 'waterville',
                       'watersmeet', 'waterford', 'watson', 'waterville_valley', 'waterbury_center', 'watertown',
                       'watonga', 'wattsville', 'watchung', 'water_view', 'waterford_works', 'watford_city', 'waterloo',
                       'watton', 'watseka', 'water_valley'],
                      ['wa'],
                      [f'zip98{code}' for code in range(0, 1000)],
                      ['s'], ['y'],
                      [0, 10000],
                      [0, 3],
                      [0, 0.1],
                      [0, 0.1],
                      [0, 0.1]
        ]
    elif 'br2000' in path:
        target_attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11',
                        'a12', 'a13', 'a14']
        pos_values = [  # total = 38k
            ['one'],  # 11134
            ['one', 'two', 'three', 'four', 'five', 'six'],  # 14268
            list(range(4, 22)),  # 14158
            ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],  # 16698
            list(range(26, 91)),  # 19906
            ['one'],  # 14417
            ['one'],  # 2833
            ['zero'],  # 12169
            ['one'],  # 267
            ['one', 'two', 'three'],  # 19024
            list(range(45, 92)),  # 15915
            ['one'],  # 36838
            list(range(25, 91)),  # 19548
            ['one'],  # 10476
        ]
    elif 'tpch' in path:
        target_attrs = ['c_custkey','c_nationkey','c_acctbal','c_mktsegment',
                        'n_name','n_regionkey','o_orderpriority','o_totalprice','o_orderstatus']
        pos_values = [  # total = 19951
            [f'custkey_{code}' for code in range(0, 110000)],
            [f'nationkey_{code}' for code in range(0, 12)],
            [-1000,8000],  
            ['automobile', 'household', 'furniture', 'building'], 
            ['vietnam','china','russia','jordan','indonesia','japan','india','iran','saudiarabia','iraq'],
            ['regionkey_2', 'regionkey_0', 'regionkey_3'],  
            ['2-high','3-medium', '5-low', '4-notspecified'],  
            [0,300000],
            ['o'],
        ]
    return target_attrs, pos_values


