import re
import math

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import hdbscan
import numpy as np
from scipy.stats import mstats

import sklearn.tree as sktree
from sklearn.tree import _tree as sktree_internal

from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import SpectralClustering

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier as skNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
            output.append("{}return '{}'".format(indent, value_map[np.argmax(
                tree_.value[node])]))

    recurse(0, 1)
    return '\n'.join(output)


def get_errors_for(classifier, dataset):
    return [
        dataset.normalized.iloc[i][classifier.get_config(
            **dataset.features.iloc[i].to_dict())]
        for i in range(dataset.normalized.shape[0])
    ]


def get_perfect_errors_for(labels, dataset):
    limited_norm = dataset.normalized[labels]
    return limited_norm.max(axis=1)


def get_tree_classifier_for(fn_name, labels, dataset):
    limited_norm = dataset.normalized[labels]
    return train_tree_classifier(
        fn_name,
        dataset.features,
        limited_norm,
        limited_norm.columns,
        min_samples_split=3,
        min_samples_leaf=3)


class TopN():
    cls_name = "Top"

    def __init__(self, dataset, n_classes):
        counts = dataset.normalized.idxmax(axis=1).value_counts()
        top_n = counts.nlargest(n=n_classes).index
        self.classes = top_n
        self.name = "{}{}".format(self.cls_name, n_classes)


class KMeans():
    cls_name = "KMeans"

    def __init__(self, dataset, n_classes):
        # Try using kmeans to work out clusters of results, so that we can pick
        # 'representatives' of each cluster to use as our kernel selection. This
        # provides classes to use in a decision tree classifier.
        kmeans = skKMeans(
            n_clusters=n_classes, random_state=0).fit(dataset.normalized)
        kernel_map = [
            dataset.normalized.columns[np.argmax(vec)]
            for vec in kmeans.cluster_centers_
        ]
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class PCAKMeans():
    cls_name = "PCAKMeans"

    def _invert_pca(self, pca, mu, data):
        n_comp = pca.n_components
        Xhat = np.dot(data[:, :n_comp], pca.components_[:n_comp, :])
        Xhat += mu
        return Xhat

    def __init__(self, dataset, n_classes):
        data = dataset.normalized.reset_index(drop=True)
        pca = PCA(n_components=25)
        pca.fit(data)
        mu = data.mean(axis=0).to_numpy()

        transformed = pca.transform(data)
        kmeans = skKMeans(
            n_clusters=n_classes, random_state=0).fit(transformed)

        centroids = self._invert_pca(pca, mu, kmeans.cluster_centers_)

        kernel_map = [data.columns[np.argmax(vec)] for vec in centroids]
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class Spectral():
    cls_name = "Spectral"

    def __init__(self, dataset, n_classes):
        cluster = SpectralClustering(
            n_clusters=n_classes, random_state=0, assign_labels='kmeans')
        cluster = cluster.fit(dataset.normalized)
        labels = cluster.labels_

        def extract_class_for(label):
            "For given label, extract all data in that label and get best kernel for them."
            data = dataset.normalized.loc[labels == label]
            if data.size == 0:
                return ''
            #return data.idxmax(axis=1).value_counts().idxmax()
            return data.mean(axis=0).idxmax()

        kernel_map = [extract_class_for(i) for i in range(0, n_classes)]
        kernel_map = [x for x in kernel_map if x != '']
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class HDBScan():
    cls_name = "HDBScan"
    PARAM_MAP = {
        15: ('l1', 3, 3),
        14: ('l1', 4, 2),
        13: ('l1', 4, 2),
        12: ('l1', 4, 2),
        11: ('l1', 4, 3),
        10: ('l1', 2, 4),
        9: ('l1', 3, 4),
        8: ('l1', 2, 5),
        7: ('l1', 5, 3),
        6: ('l1', 6, 3),
        5: ('l1', 9, 1),
        4: ('l1', 2, 13),
    }

    def __init__(self, dataset, n_classes):
        # HDBScan is a better clustering algorithm that may give a better set of
        # representatives.
        m, c, s = HDBScan.PARAM_MAP[n_classes]
        clusterer = hdbscan.HDBSCAN(
            metric=m, min_cluster_size=c, min_samples=s)
        clusterer.fit(dataset.normalized)
        assert clusterer.labels_.max() <= n_classes

        # For each cluster, choose a representative that gives the best overall
        # performance for the class exemplars.
        chosen_labels = [
            np.argmax(mstats.gmean(x, axis=0)) for x in clusterer.exemplars_
        ]
        kernel_map = [dataset.normalized.columns[i] for i in chosen_labels]
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class DecisionTree():
    cls_name = "DecisionTree"

    def __init__(self, dataset, n_classes):
        # Can use a decision tree regressor to try to model the full data set without
        # pruning, by setting the maximum number of leaf nodes.
        model = sktree.DecisionTreeRegressor(
            random_state=0,
            min_samples_split=3,
            min_samples_leaf=3,
            max_leaf_nodes=n_classes,
        )
        model = model.fit(dataset.features, dataset.normalized)
        self.classes = [
            dataset.normalized.columns[np.argmax(vec)]
            for vec in model.tree_.value
        ]
        self.name = "{}{}".format(self.cls_name, n_classes)


resnet = load_cached('resnet_batch1_matmuls_amd_out.csv')
vgg = load_cached('vgg_batch1_matmuls_amd_out.csv')
mobilenet = load_cached('mobilenet_batch1_matmuls_amd_out.csv')

res_vgg = combine(resnet, vgg)
all_data = combine(res_vgg, mobilenet)


def print_component_numbers_from_pca(values):
    pca = PCA()
    pca.fit(values)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print("Number components for 80% variance: ", np.argmax(cumsum > 0.8) + 1)
    print("Number components for 90% variance: ", np.argmax(cumsum > 0.9) + 1)
    print("Number components for 95% variance: ", np.argmax(cumsum > 0.95) + 1)


print_component_numbers_from_pca(all_data.normalized)

MODELS = [TopN, DecisionTree, KMeans, PCAKMeans, HDBScan, Spectral]

N_CLASSES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

feat_train, feat_test, norm_train, norm_test, val_train, val_test = train_test_split(
    all_data.features,
    all_data.normalized,
    all_data.values,
    test_size=0.2,
    random_state=42)


def get_dataset_from(feat, norm, val):
    return DataSet(
        feat.reset_index(drop=True),
        norm.reset_index(drop=True),
        val.reset_index(drop=True))


train_dataset = get_dataset_from(feat_train, norm_train, val_train)
test_dataset = get_dataset_from(feat_test, norm_test, val_test)

chosen_labels = {}


def compare_train(train, test):
    for n_c in N_CLASSES:
        for model in MODELS:
            m = model(train, n_c)
            labels = m.classes
            error = geom_mean(get_perfect_errors_for(labels, test))
            chosen_labels[m.name] = labels
            print("{},{},{}".format(m.__class__.__name__, n_c, error))


print("--- Kernel pruning performance ---")
compare_train(train_dataset, test_dataset)


def get_targets_for_given_configs(dataset, labels):
    limited_norm = dataset.normalized[labels]
    return limited_norm.idxmax(axis=1)


def augment_features(feats):

    def get_feats(m, k, n, b):
        return (m, k, n, b, m * n, b * m * n, k * n, k * m, m / k, n / k,
                m * n / k)

    feats['mn'] = feats['m'] * feats['n']
    feats['bmn'] = feats['batch'] * feats['m'] * feats['n']
    feats['kn'] = feats['k'] * feats['n']
    feats['km'] = feats['k'] * feats['m']
    feats['mpk'] = feats['m'] / feats['k']
    feats['npk'] = feats['n'] / feats['k']
    feats['mnpk'] = feats['m'] * feats['n'] / feats['k']

    return feats, get_feats


_, get_feats = augment_features(train_dataset.features)


class GenModel():
    def __init__(self, model, classifier):
        labels = chosen_labels[model]
        x_labels = get_targets_for_given_configs(train_dataset, labels)
        self.model = classifier.fit(train_dataset.features, x_labels)

    def get_config(self, m, k, n, batch):
        return self.model.predict(np.asarray([get_feats(m, k, n, batch)]))


def get_classifier_performance(model):
    errors = [
        test_dataset.normalized.iloc[i][model.get_config(**test_dataset.features.iloc[i].to_dict())][0]
        for i in range(test_dataset.normalized.shape[0])
    ]
    return geom_mean(errors), errors


CLASSIFIER_MOD = [DecisionTree]
CLASSIFIER_CLS = [5, 6, 8, 15]
CLASSIFIERS = [
    ('DecisionTree', sktree.DecisionTreeClassifier(random_state=0)),
    ('LimitedDecisionTree', sktree.DecisionTreeClassifier(
        random_state=0,
        max_features=4,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=3)),
    ('1NearestNeighbor', skNeighborsClassifier(n_neighbors=1)),
    ('3NearestNeighbor', skNeighborsClassifier(n_neighbors=3)),
    ('7NearestNeighbor', skNeighborsClassifier(n_neighbors=7)),
    ('LinearSVM', SVC(kernel="linear", C=0.025, cache_size=5000)),
    ('RadialSVM', SVC(gamma=2, C=1, cache_size=5000)),
    ('RandomForest', RandomForestClassifier(random_state=0)),
]


def compare_classifiers():
    for n_c in CLASSIFIER_CLS:
        for model in CLASSIFIER_MOD:
            classes = '{}{}'.format(model.cls_name, n_c)

            for cls_name, classifier in CLASSIFIERS:
                m = GenModel(classes, classifier)
                error, errors = get_classifier_performance(m)
                print(classes, cls_name, error)


print("--- Classifier performance ---")
compare_classifiers()
