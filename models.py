import itertools

import dataset
import utils

import numpy as np
import hdbscan

from scipy.stats import mstats
import sklearn.tree as skTree
from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

np.seterr(all='ignore')

def best_of_single(x):
    return np.argmax(x)

def best_of_mean(x):
    return best_of_single(x.mean(axis=0))

def best_of_gmean(x):
    return best_of_single(mstats.gmean(x, axis=0))


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
        # 'representatives' of each cluster to use as our kernel selection.
        # This provides classes to use in a decision tree classifier.
        kmeans = skKMeans(n_clusters=n_classes,
                          random_state=0).fit(dataset.normalized)
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
        kmeans = skKMeans(n_clusters=n_classes,
                          random_state=0).fit(transformed)

        centroids = self._invert_pca(pca, mu, kmeans.cluster_centers_)

        kernel_map = [data.columns[np.argmax(vec)] for vec in centroids]
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class Spectral():
    cls_name = "Spectral"

    def __init__(self, dataset, n_classes):
        cluster = SpectralClustering(n_clusters=n_classes,
                                     random_state=0,
                                     assign_labels='kmeans')
        cluster = cluster.fit(dataset.normalized)
        labels = cluster.labels_

        def extract_class_for(label):
            """
            For given label, extract all data in that label and get best kernel
            for them.
            """
            data = dataset.normalized.loc[labels == label]
            if data.size == 0:
                return ''
            return data.mean(axis=0).idxmax()

        kernel_map = [extract_class_for(i) for i in range(0, n_classes)]
        kernel_map = [x for x in kernel_map if x != '']
        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)


class PCASpectral():
    cls_name = "PCASpectral"

    def __init__(self, dataset, n_classes):
        data = dataset.normalized.reset_index(drop=True)
        pca = PCA(n_components=25)
        pca.fit(data)
        mu = data.mean(axis=0).to_numpy()

        transformed = pca.transform(data)
        cluster = SpectralClustering(n_clusters=n_classes,
                                     random_state=0,
                                     assign_labels='kmeans')
        cluster = cluster.fit(transformed)
        labels = cluster.labels_

        def extract_class_for(label):
            """
            For given label, extract all data in that label and get best kernel
            for them.
            """
            data = dataset.normalized.loc[labels == label]
            if data.size == 0:
                return ''
            return data.mean(axis=0).idxmax()

        kernel_map = [extract_class_for(i) for i in range(0, n_classes)]
        kernel_map = [x for x in kernel_map if x != '']

        self.classes = kernel_map
        self.name = "{}{}".format(self.cls_name, n_classes)

CLUSTERS = range(2, 10)
SAMPLES = range(1, 15)
hdbscan_cache = {}
def compute_hdbscan(data):
    global hdbscan_cache
    if 4 in hdbscan_cache:
        return hdbscan_cache

    # Recompute raw dataset scales, as the normalization may not be scale
    scales = data.values.div(data.values.max(axis=1), axis=0)
    test_data = dataset.from_values(data.features, scales, data.values)

    # Build map from number of classes to configs
    ranges = {}
    for c, s in itertools.product(CLUSTERS, SAMPLES):
        clusterer = hdbscan.HDBSCAN(metric='l2',
                                    min_cluster_size=c,
                                    min_samples=s)
        clusterer.fit(data.normalized)
        # Labels are 0-indexed
        n_classes = clusterer.labels_.max() + 1
        chosen_labels = [
            np.argmax(mstats.gmean(x, axis=0)) for x in clusterer.exemplars_
        ]
        kernel_map = [data.normalized.columns[i] for i in chosen_labels]
        err = utils.geom_mean(utils.get_perfect_errors_for(kernel_map, test_data))
        #print("hdbscan {} classes for {}, {}. err {}".format(n_classes, c, s, err))
        #print('\n'.join(kernel_map))
        if n_classes in ranges.keys():
            ranges[n_classes] += [(c, s, err)]
        else:
            ranges[n_classes] = [(c, s, err)]

    for i in sorted(ranges.keys()):
        print("hdbscan: {} -> {}".format(i, ranges[i]))

    # Scan through map to get best trained config for each number of classes
    configs = {0 : (15, 15)}
    for i in range(1, 16):
        if i in ranges:
            m = 0
            for c, s, e in ranges[i]:
                if e > m:
                    configs[i] = (c, s)
                    m = e
        else:
            configs[i] = configs[i-1]

    hdbscan_cache = configs
    return configs

def reset_hdbscan_cache():
    global hdbscan_cache
    hdbscan_cache = {}

class HDBScan():
    cls_name = "HDBScan"

    def __init__(self, dataset, n_classes):
        # HDBScan is a better clustering algorithm that may give a better set
        # of representatives.
        c, s = compute_hdbscan(dataset)[n_classes]
        clusterer = hdbscan.HDBSCAN(metric='l2',
                                    min_cluster_size=c,
                                    min_samples=s)
        clusterer.fit(dataset.normalized)

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
        # Can use a decision tree regressor to try to model the full data set
        # without pruning, by setting the maximum number of leaf nodes.
        model = skTree.DecisionTreeRegressor(
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
