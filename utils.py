import numpy as np

from scipy.stats import mstats
from sklearn.decomposition import PCA
from sklearn.tree import _tree as sktree_internal


def geom_mean(values):
    ''' Get the geometric mean of the given values. '''
    return mstats.gmean(values)


def tree_to_code(fn_name, tree, feature_names, value_map):
    '''
    Convert a descision tree to a python function.

    The function will be returned as a string containing the full definition.

    Example output for tree_to_code('fn', tree, ('a', 'b', 'c'), ...):
        def fn(a, b, c):
          if a < 10:
            return 'x'
          elif b < 1:
            return 'y'
          else
            return 'z'
    '''
    tree_ = tree.tree_
    feature_name = [
        feature_names[i]
        if i != sktree_internal.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    output = ["def {}({}):".format(fn_name, ", ".join(feature_names))]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != sktree_internal.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            output.append("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            output.append("{}else:  # if {} > {}".format(
                indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            output.append("{}return '{}'".format(
                indent, value_map[np.argmax(tree_.value[node])]))

    recurse(0, 1)
    return '\n'.join(output)


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
