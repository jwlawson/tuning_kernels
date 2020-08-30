import argparse
import itertools
import math
import sys

import numpy as np

import sklearn.tree as sktree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as skNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import dataset
import models
import utils


def print_PCA_components(data):
    pca_variance = utils.cumulative_pca_variance(data.normalized)
    print("Number components for 80% variance: {}".format(
        np.argmax(pca_variance > 0.8) + 1))
    print("Number components for 90% variance: {}".format(
        np.argmax(pca_variance > 0.9) + 1))
    print("Number components for 95% variance: {}".format(
        np.argmax(pca_variance > 0.95) + 1))


def split_dataset(all_data):
    (feat_train, feat_test, norm_train, norm_test, val_train,
     val_test) = train_test_split(all_data.features,
                                  all_data.normalized,
                                  all_data.values,
                                  test_size=0.2,
                                  random_state=42)

    train_dataset = dataset.from_values(feat_train, norm_train, val_train)
    test_dataset = dataset.from_values(feat_test, norm_test, val_test)
    return train_dataset, test_dataset


def normalize_data(data, norm):
    if norm == 'rawcutoff':
        return dataset.DataSet(
            features=data.features,
            normalized=data.normalized.applymap(lambda x: x if x > 0.9 else 0),
            values=data.values,
        )
    elif norm == 'cutoff':
        return dataset.DataSet(
            features=data.features,
            normalized=data.normalized.applymap(lambda x: (x - 0.9) / 0.1
                                                if x > 0.9 else 0),
            values=data.values,
        )
    elif norm == 'sigmoid':
        return dataset.DataSet(
            features=data.features,
            normalized=data.normalized.applymap(
                lambda x: (1 + math.exp(50 * (0.85 - x)))**-1),
            values=data.values,
        )
    return data


MODELS = [
    models.TopN,
    models.DecisionTree,
    models.KMeans,
    models.PCAKMeans,
    models.Spectral,
    models.HDBScan,
]

N_CLASSES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

chosen_labels = {}


def compare_train(train, test, norm, out):
    global chosen_labels
    for n_c, model in itertools.product(N_CLASSES, MODELS):
        m = model(train, n_c)
        labels = m.classes
        error = utils.geom_mean(utils.get_perfect_errors_for(labels, test))
        chosen_labels[m.name] = labels
        print("{},{},{},{}".format(m.__class__.__name__, n_c, error, norm),
              file=out)


def get_targets_for_given_configs(dataset, labels):
    limited_norm = dataset.normalized[labels]
    return limited_norm.idxmax(axis=1)


def augment_features(feats):
    def get_feats(m, k, n, b):
        return (m, k, n, b, m * n, b * m * n, k * n, k * m, m / k, n / k,
                m * n / k)

    feats['m*n'] = feats['m'] * feats['n']
    feats['b*m*n'] = feats['batch'] * feats['m'] * feats['n']
    feats['k*n'] = feats['k'] * feats['n']
    feats['k*m'] = feats['k'] * feats['m']
    feats['m/k'] = feats['m'] / feats['k']
    feats['n/k'] = feats['n'] / feats['k']
    feats['m*n/k'] = feats['m'] * feats['n'] / feats['k']

    return feats, get_feats


class GenModel():
    def __init__(self, data, model, classifier, get_feats):
        labels = chosen_labels[model]
        x_labels = get_targets_for_given_configs(data, labels)
        self.model = classifier.fit(data.features, x_labels)
        self.get_feats = get_feats

    def get_config(self, m, k, n, batch):
        return self.model.predict(np.asarray([self.get_feats(m, k, n, batch)]))


def get_classifier_performance(data, model):
    errors = [
        data.normalized.iloc[i][model.get_config(
            **data.features.iloc[i].to_dict())][0]
        for i in range(data.normalized.shape[0])
    ]
    return utils.geom_mean(errors), errors


CLASSIFIER_MOD = [
    #models.DecisionTree,
    models.PCAKMeans,
    #models.HDBScan,
]
CLASSIFIER_CLS = [5, 6, 8, 15]
CLASSIFIERS = [
    ('DecisionTreeA', sktree.DecisionTreeClassifier(random_state=0)),
    ('DecisionTreeB',
     sktree.DecisionTreeClassifier(random_state=0,
                                   max_depth=6,
                                   min_samples_split=4,
                                   min_samples_leaf=3)),
    ('DecisionTreeC',
     sktree.DecisionTreeClassifier(random_state=0,
                                   max_depth=3,
                                   min_samples_split=5,
                                   min_samples_leaf=4)),
    ('1NearestNeighbor', skNeighborsClassifier(n_neighbors=1)),
    ('3NearestNeighbor', skNeighborsClassifier(n_neighbors=3)),
    ('7NearestNeighbor', skNeighborsClassifier(n_neighbors=7)),
    ('LinearSVM', LinearSVC(C=0.25, random_state=0, max_iter=5000)),
    ('RadialSVM', SVC(gamma=2, C=0.25, cache_size=5000, random_state=0)),
    ('RandomForest', RandomForestClassifier(random_state=0)),
    ('MLP',
     MLPClassifier(solver='lbfgs',
                   alpha=1e-5,
                   hidden_layer_sizes=(16, 16),
                   random_state=0)),
]


def compare_classifiers(train, test, norm, get_feats, out):
    for n_c, model in itertools.product(CLASSIFIER_CLS, CLASSIFIER_MOD):
        for cls_name, classifier in CLASSIFIERS:
            classes = '{}{}'.format(model.cls_name, n_c)
            m = GenModel(train, classes, classifier, get_feats)
            error, errors = get_classifier_performance(test, m)
            print('{},{},{:.2f},{}'.format(classes, cls_name, error * 100, norm),
                  file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--norm',
                        nargs='*',
                        default=['scale', 'cutoff', 'rawcutoff', 'sigmoid'])
    parser.add_argument('--prune-out',
                        nargs='?',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('--class-out',
                        nargs='?',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    args = parser.parse_args()

    rawdata = dataset.load_cached(args.filename)
    print_PCA_components(rawdata)
    train_dataset, test_dataset = split_dataset(rawdata)

    for norm in args.norm:
        data = normalize_data(train_dataset, norm)

        compare_train(data, test_dataset, norm, args.prune_out)
        _, get_feats = augment_features(data.features)
        compare_classifiers(data, test_dataset, norm, get_feats,
                            args.class_out)
        # Hack to remove HDBScan cache, as it doesn't know about different
        # files or normalization schemes
        models.reset_hdbscan_cache()


if __name__ == '__main__':
    main()
