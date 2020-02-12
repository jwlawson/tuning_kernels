import itertools

import numpy as np

import sklearn.tree as sktree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as skNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import dataset
import models
import utils

resnet = dataset.load_cached('resnet_batch1_matmuls_amd_out.csv')
vgg = dataset.load_cached('vgg_batch1_matmuls_amd_out.csv')
mobilenet = dataset.load_cached('mobilenet_batch1_matmuls_amd_out.csv')

res_vgg = dataset.combine(resnet, vgg)
all_data = dataset.combine(res_vgg, mobilenet)

pca_variance = utils.cumulative_pca_variance(all_data.normalized)
print("Number components for 80% variance: {}".format(
    np.argmax(pca_variance > 0.8) + 1))
print("Number components for 90% variance: {}".format(
    np.argmax(pca_variance > 0.9) + 1))
print("Number components for 95% variance: {}".format(
    np.argmax(pca_variance > 0.95) + 1))

MODELS = [
    models.TopN, models.DecisionTree, models.KMeans, models.PCAKMeans,
    models.HDBScan, models.Spectral
]

N_CLASSES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

(feat_train, feat_test, norm_train, norm_test, val_train,
 val_test) = train_test_split(all_data.features,
                              all_data.normalized,
                              all_data.values,
                              test_size=0.2,
                              random_state=42)

train_dataset = dataset.from_values(feat_train, norm_train, val_train)
test_dataset = dataset.from_values(feat_test, norm_test, val_test)

chosen_labels = {}


def compare_train(train, test):
    for n_c, model in itertools.product(N_CLASSES, MODELS):
        m = model(train, n_c)
        labels = m.classes
        error = utils.geom_mean(utils.get_perfect_errors_for(labels, test))
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

    feats['m*n'] = feats['m'] * feats['n']
    feats['b*m*n'] = feats['batch'] * feats['m'] * feats['n']
    feats['k*n'] = feats['k'] * feats['n']
    feats['k*m'] = feats['k'] * feats['m']
    feats['m/k'] = feats['m'] / feats['k']
    feats['n/k'] = feats['n'] / feats['k']
    feats['m*n/k'] = feats['m'] * feats['n'] / feats['k']

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
        test_dataset.normalized.iloc[i][model.get_config(
            **test_dataset.features.iloc[i].to_dict())][0]
        for i in range(test_dataset.normalized.shape[0])
    ]
    return utils.geom_mean(errors), errors


CLASSIFIER_MOD = [models.DecisionTree]
CLASSIFIER_CLS = [5, 6, 8, 15]
CLASSIFIERS = [
    ('DecisionTree', sktree.DecisionTreeClassifier(random_state=0)),
    ('LimitedDecisionTree',
     sktree.DecisionTreeClassifier(random_state=0,
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
    for n_c, model in itertools.product(CLASSIFIER_CLS, CLASSIFIER_MOD):
        for cls_name, classifier in CLASSIFIERS:
            classes = '{}{}'.format(model.cls_name, n_c)
            m = GenModel(classes, classifier)
            error, errors = get_classifier_performance(m)
            print(classes, cls_name, error)


print("--- Classifier performance ---")
compare_classifiers()
