import re

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import hdbscan
import numpy as np
from scipy.stats import mstats

import sklearn.tree as sktree
from sklearn.tree import _tree as sktree_internal
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from load import *


def geom_mean(norm_preds):
    return mstats.gmean(norm_preds)
    #return np.linalg.norm([1 - x for x in norm_preds])


def tree_to_code(fn_name, tree, feature_names, value_map):
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


def train_tree_regressor(fn_name, feats, labels, kernel_map, **kwargs):
    model = sktree.DecisionTreeRegressor(random_state=0, **kwargs)
    model = model.fit(feats, labels)
    return tree_to_code(fn_name, model, ['m', 'k', 'n', 'batch'],
                        kernel_map), model


def train_tree_classifier(fn_name, feats, labels, kernel_map, **kwargs):
    model = sktree.DecisionTreeClassifier(random_state=0, **kwargs)
    model = model.fit(feats, labels)
    return tree_to_code(fn_name, model, ['m', 'k', 'n', 'batch'], kernel_map)


def get_errors_for(classifier, dataset):
    return [
        dataset.normalized.iloc[i][classifier.get_config(
            **dataset.features.iloc[i].to_dict())]
        for i in range(dataset.normalized.shape[0])
    ]


class Top1():
    name = "Top1"

    def __init__(self, dataset):
        counts = dataset.normalized.idxmax(axis=1).value_counts()
        self.best_config = counts.idxmax()
        self.classes = [self.best_config]
        print("top1: ", self.classes)

    def get_config(self, m, k, n, batch):
        return self.best_config


class Top8():
    name = "Top8"

    def __init__(self, dataset):
        counts = dataset.normalized.idxmax(axis=1).value_counts()
        top_8 = counts.nlargest(n=8).index
        top8_labels = [
            dataset.normalized.iloc[i][top_8].idxmax()
            for i in range(dataset.normalized.shape[0])
        ]
        top8_tree = train_tree_classifier(
            'top8_tree_fn',
            dataset.features,
            top8_labels,
            top_8,
            min_samples_split=3,
            min_samples_leaf=3,
        )
        exec(top8_tree, globals())
        self.config_fn = top8_tree_fn
        self.classes = top_8
        print("top8: ", self.classes)

    def get_config(self, m, k, n, batch):
        return self.config_fn(m, k, n, batch)


class KMeansTree():
    name = "KMeansTree"

    def __init__(self, dataset):
        # Try using kmeans to work out clusters of results, so that we can pick
        # 'representatives' of each cluster to use as our kernel selection. This
        # provides classes to use in a decision tree classifier.
        kmeans = KMeans(n_clusters=8, random_state=0).fit(dataset.normalized)
        kernel_map = [
            dataset.normalized.columns[np.argmax(vec)]
            for vec in kmeans.cluster_centers_
        ]
        kmeans_tree = train_tree_classifier(
            'kmeans_tree_fn',
            dataset.features,
            kmeans.labels_,
            kernel_map,
            min_samples_split=5,
            min_samples_leaf=5,
        )
        exec(kmeans_tree, globals())
        self.config_fn = kmeans_tree_fn
        self.classes = kernel_map
        print("kmeans: ", self.classes)

    def get_config(self, m, k, n, batch):
        return self.config_fn(m, k, n, batch)


class HDBScanTree():
    name = "HDBScanTree"

    def __init__(self, dataset):
        # HDBScan is a better clustering algorithm that may give a better set of
        # representatives.
        clusterer = hdbscan.HDBSCAN(metric='l2',
                                    min_cluster_size=2,
                                    min_samples=7)
        clusterer.fit(dataset.normalized)
        assert clusterer.labels_.max() < 8

        # For each cluster, choose a representative that gives the best overall
        # performance for the class exemplars.
        chosen_labels = [
            np.argmax(mstats.gmean(x, axis=0)) for x in clusterer.exemplars_
        ]
        scan_labels = clusterer.labels_

        # For any outliers that were not already classified, choose the class
        # that has the best performance.
        odd_ones = dataset.normalized[scan_labels < 1]
        for idx, values in odd_ones.iterrows():
            chosen_values = values[chosen_labels].reset_index(drop=True)
            scan_labels[idx] = chosen_values.idxmax()

        kernel_map = [dataset.normalized.columns[i] for i in chosen_labels]
        scan_tree = train_tree_classifier(
            'scan_tree_fn',
            dataset.features,
            scan_labels,
            kernel_map,
            min_samples_split=3,
            min_samples_leaf=2,
        )
        exec(scan_tree, globals())
        self.config_fn = scan_tree_fn
        self.classes = kernel_map
        print("hdbscan: ", kernel_map)

    def get_config(self, m, k, n, batch):
        return self.config_fn(m, k, n, batch)


class DecisionTree():
    name = "DecisionTree"

    def __init__(self, dataset):
        # Can use a decision tree regressor to try to model the full data set without
        # pruning, by setting the maximum number of leaf nodes.
        reg_tree, model = train_tree_regressor(
            'reg_tree_fn',
            dataset.features,
            dataset.normalized,
            dataset.normalized.columns,
            min_samples_split=3,
            min_samples_leaf=3,
            max_leaf_nodes=8,
        )

        exec(reg_tree, globals())
        self.config_fn = reg_tree_fn
        self.classes = re.findall(r'return \'(.*)\'', reg_tree)
        print("dec tree: ", self.classes)

    def get_config(self, m, k, n, batch):
        return self.config_fn(m, k, n, batch)


resnet = load_cached('resnet_batch1_matmuls_amd_out.csv')
vgg = load_cached('vgg_batch1_matmuls_amd_out.csv')
mobilenet = load_cached('mobilenet_batch1_matmuls_amd_out.csv')

res_vgg = combine(resnet, vgg)
all_data = combine(res_vgg, mobilenet)

MODELS = [Top1, Top8, DecisionTree, KMeansTree, HDBScanTree]

feat_train, feat_test, norm_train, norm_test, val_train, val_test = train_test_split(
    all_data.features, all_data.normalized, all_data.values, test_size=0.2)
print(feat_train)
print(feat_test)


def compare_train(train, test):
    for model in MODELS:
        m = model(train)
        error = geom_mean(get_errors_for(m, test))
        print("{} error: {}".format(model.name, error))


compare_train(
    DataSet(feat_train.reset_index(drop=True),
            norm_train.reset_index(drop=True),
            val_train.reset_index(drop=True)),
    DataSet(feat_test.reset_index(drop=True), norm_test.reset_index(drop=True),
            val_test.reset_index(drop=True)))
