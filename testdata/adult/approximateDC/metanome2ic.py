from synthesizer.evaluation import _validate_dc_vio


def parse(raw_path='./metanome.out', out_path='./adult150.ic'):
    """
    Parse the output of approximate DCs from Metanome to our DC format
    :param raw_path the raw output from metanome project. Default is on file.
    :param out_path location to save the DCs
    """
    ics = []
    ops = {'=': 'EQ', '!=': 'IQ', '<=': 'LTE', '<': 'LT', '>=': 'GTE', '>': 'GT'}
    f = open(raw_path, "r")
    for idx, line in enumerate(f):
        line = line.replace('t1', 't2').replace('t0', 't1').rstrip()
        conds = line.split(" and ")
        ic = 't1&t2'
        for cond in conds:
            l = len(cond)
            i = l//2
            if l % 2 == 0:
                op = cond[i-1:i+1]
            else:
                op = cond[i]

            ic += '&'
            ic += ops[op]
            preds = cond.split(op)
            ic += '(' + preds[0] + ',' + preds[1] + ')'

        ics.append(ic)
        print(f'{idx}\t{ic}')

    with open(out_path, 'w') as writer:
        for ic in ics:
            writer.write(ic+'\n')


if __name__ == '__main__':

    path_ic = './adult150.ic'
    path_adult = './adult.csv'

    n_vios, n_pairs = _validate_dc_vio(path_adult, path_ic)
    for i in range(len(n_vios)):
        n_vio = n_vios[i]
        n_pair = n_pairs[i]
        print(f"{i}\t{n_vio}\t{n_pair}\t{'{:.2f}'.format(n_vio/n_pair)}")