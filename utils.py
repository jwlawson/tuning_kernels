import math

import numpy as np

from scipy.stats import mstats
from sklearn.decomposition import PCA
from sklearn.tree import _tree as sktree_internal


def geom_mean(values):
    ''' Get the geometric mean of the given values. '''
    return mstats.gmean(values)


def config_to_launch(config):
    tiles, wgs = config.split(':')
    tiles = tiles.split('_')
    wgs = wgs.split('_')
    return 'LAUNCH({0[0]}, {0[1]}, {0[2]}, {1[0]}, {1[1]}, {1[2]})'.format(tiles, wgs)


def tree_to_code(fn_name, tree, feature_names, value_map):
    '''
    Convert a descision tree to a python function.

    The function will be returned as a string containing the full definition.

    Example output for tree_to_code('fn', tree, ('a', 'b', 'c'), ...):
        def fn(a, b, c):
          if a < 10:
            return 'x'
          else:
            if b < 1:
              return 'y'
            else:
              return 'z'
    '''
    tree_ = tree.tree_
    feature_name = [
        feature_names[i]
        if i != sktree_internal.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    tree_string = ["def {}({}):".format(fn_name, ", ".join(feature_names))]

    def indent(strings):
        return ["  {}".format(s) for s in strings]

    def recurse(node):
        if tree_.feature[node] != sktree_internal.TREE_UNDEFINED:
            name = feature_name[node]
            if '/' in name:
                name = '(float){}'.format(name)
            threshold = tree_.threshold[node]
            if threshold == math.floor(threshold):
                threshold = int(threshold)
            left_output = recurse(tree_.children_left[node])
            right_output = recurse(tree_.children_right[node])

            if left_output == right_output:
                # Return early to avoid extra indentation
                return left_output
            else:
                output = [
                    "if ({} <= {}) {{".format(name, threshold),
                    *left_output,
                    "} else {",
                    *right_output,
                    "}",
                ]
        else:
            return_value = value_map[np.argmax(tree_.value[node])]
            return_value = config_to_launch(return_value)
            output = ["return {};".format(return_value)]

        return indent(output)

    tree_string += recurse(0)
    return '\n'.join(tree_string)


def get_errors_for(classifier, dataset):
    '''
    Extract the error between the the classifier output and the optimal results
    in the dataset.

    Returns a list of errors.
    '''
    return [
        dataset.normalized.iloc[i][classifier.get_config(
            **dataset.features.iloc[i].to_dict())]
        for i in range(dataset.normalized.shape[0])
    ]


def get_perfect_errors_for(kernels, dataset):
    '''
    Get a list of the maximum achievable performance given a subset of kernels.
    '''
    limited_norm = dataset.normalized[kernels]
    return limited_norm.max(axis=1)


def cumulative_pca_variance(values):
    pca = PCA()
    pca.fit(values)
    return np.cumsum(pca.explained_variance_ratio_)
