import pandas as pd
import numpy as np

import pickle
import os

from collections import namedtuple

DataSet = namedtuple('DataSet', ['features', 'normalized', 'values'])

def load_from_csv(filename):
    data = pd.read_csv(filename)
    data['config'] = data.apply(lambda x: '{}_{}_{}:{}_{}_{}'.format(
        x.row_tile, x.acc_tile, x.col_tile, x.wg_a, x.wg_b, x.wg_c),
                                axis=1)
    pivot = data.pivot_table(index=['m', 'k', 'n', 'batch'],
                             columns=['config'],
                             values='items_per_second').reset_index()
    features = pivot[['m', 'k', 'n', 'batch']]
    values = pivot.drop(['m', 'k', 'n', 'batch'], axis=1)
    normalized = values.div(values.max(axis=1), axis=0)
    return features, normalized, values


def save_to_pickle(filename, features, normalized, values):
    df_dict = {
        'features': features,
        'normalized': normalized,
        'values': values,
    }
    with open(filename, 'wb') as f:
        pickle.dump(df_dict, f)


def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        dfs = pickle.load(f)
    return DataSet(dfs['features'], dfs['normalized'], dfs['values'])


def load_cached(filename):
    root_file, _ = os.path.splitext(filename)
    pickle_file = root_file + '.pkl'
    pickle_exists = os.path.exists(pickle_file)
    if pickle_exists:
        return load_from_pickle(pickle_file)
    else:
        a, b, c = load_from_csv(filename)
        save_to_pickle(pickle_file, a, b, c)
    return DataSet(a, b, c)


def combine(dataset1, dataset2):
    feat = pd.concat([dataset1.features, dataset2.features]).reset_index(drop=True)
    norm = pd.concat([dataset1.normalized, dataset2.normalized]).reset_index(drop=True)
    values = pd.concat([dataset1.values, dataset2.values]).reset_index(drop=True)
    return DataSet(feat, norm, values)


