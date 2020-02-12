import numpy as np
import hdbscan

from scipy.stats import mstats
import sklearn.tree as skTree
from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA


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
        # HDBScan is a better clustering algorithm that may give a better set
        # of representatives.
        m, c, s = HDBScan.PARAM_MAP[n_classes]
        clusterer = hdbscan.HDBSCAN(metric=m,
                                    min_cluster_size=c,
                                    min_samples=s)
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
