#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import metrics


def compare_k_AggClustering(k_list, X):
    # to find the best k number of clusters
    X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:

        clusterer = AgglomerativeClustering(n_clusters=p, linkage="average")

        clusterer.fit(X)
        # The higher (up to 1) the better
        s = round(metrics.silhouette_score(X, clusterer.labels_), 4)

        silhouette_list.append(s)

    # The higher (up to 1) the better
    key = silhouette_list.index(max(silhouette_list))

    k = k_list.__getitem__(key)

    print("Best silhouette =", max(silhouette_list), " for k=", k)

    return k


def compare_k_means(k_list, X):
    # to find the best k number of clusters

    X = X.select_dtypes(['number']).dropna()
    # Run clustering with different k and check the metrics
    silhouette_list = []

    for p in k_list:

        clusterer = KMeans(n_clusters=p, n_jobs=4)

        clusterer.fit(X)

        # The higher (up to 1) the better
        s = round(metrics.silhouette_score(X, clusterer.labels_), 4)

        silhouette_list.append(s)

    # The higher (up to 1) the better
    key = silhouette_list.index(max(silhouette_list))

    k = k_list.__getitem__(key)

    print("Best silhouette =", max(silhouette_list), " for k=", k)

    return k


class Clusterer():
    """
    Clustering task using a particular method

    Parameters
    ----------
    *  dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'KMEANS' ; The choice for the clustering method:
        - 'KMEANS', 'HCA' and 'DBSCAN'

    * metric: str, default = 'euclidean'

    * verbose: Boolean,  default = 'False' otherwise display the list of
        cluster labels resulting from clustering
    """

    def __init__(self, dataset,  strategy='KMEANS', metric='euclidean',
                 verbose=False):

        self.dataset = dataset

        self.strategy = strategy

        self.metric = metric

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'metric': self.metric,

                'verbose': self.verbose}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for clusterer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`clusterer.get_params().keys()`")
            else:

                setattr(self, k, v)

    # K-Means Clustering
    def KMEANS_clustering(self, dataset):

        X = dataset.select_dtypes(['number']).dropna()

        if X.shape[1] < 1:

            silhouette = None

            data = dataset

            print("Error: There are too few observations")

        else:

            n_clusters = [2, 3, 4, 5]

            k = compare_k_means(n_clusters, dataset)

            clusterer_final = KMeans(n_clusters=k, n_jobs=-1)

            clusterer_final = clusterer_final.fit(X)

            silhouette = round(metrics.silhouette_score(
                X, clusterer_final.labels_), 4)

            data = X.assign(cluster_ID=clusterer_final.labels_)

        print("Quality of clustering", silhouette)

        if self.verbose:

            print("Labels distribution:")

            print(data['cluster_ID'].value_counts())

        return silhouette, data

    # metrics "cosine", "euclidean", "cityblock"
    def HCA_clustering(self, dataset, metric):

        X = dataset.select_dtypes(['number']).dropna()

        if X.shape[1] < 1:

            silhouette = None

            data = dataset

            print("Error: There are too few observations")

        else:

            n_clusters = [2, 3, 4, 5]

            k = compare_k_AggClustering(n_clusters, dataset)

            clusterer_final = AgglomerativeClustering(
                n_clusters=k, linkage="average", affinity=metric)

            clusterer_final = clusterer_final.fit(X)

            silhouette = round(metrics.silhouette_score(
                X, clusterer_final.labels_), 4)

            data = X.assign(cluster_ID=clusterer_final.labels_)

        print("Quality of clustering", silhouette)

        if self.verbose:

            print("Labels distribution:")

            print(data['cluster_ID'].value_counts())

        return silhouette, data

    def DBSCAN_clustering(self, dataset):

        X = dataset.select_dtypes(['number']).dropna()

        if X.shape[1] < 1:

            silhouette = None

            data = dataset

            print("Error: There are too few observations")

        else:

            clusterer_final = DBSCAN(eps=0.1)

            clusterer_final = clusterer_final.fit(X)

            if len(np.unique(clusterer_final.labels_)) > 1:

                silhouette = round(metrics.silhouette_score(
                    X, clusterer_final.labels_), 4)

                data = X.assign(cluster_ID=clusterer_final.labels_)

            else:

                silhouette = None

                data = X.assign(cluster_ID=clusterer_final.labels_)

                print("Silhouette cannot be computed because only one "
                      "cluster/label")

        print("Quality of clustering", silhouette)

        if self.verbose and (silhouette is not None):

            print("Labels distribution:")

            print(data['cluster_ID'].value_counts())

        return silhouette, data

    def transform(self):

        start_time = time.time()

        clustd = self.dataset

        print()

        print(">>Clustering task")

        print("Note: The clustering is applied on the training dataset only.")

        d = self.dataset['train']

        if (self.strategy == "KMEANS"):

            dn = self.KMEANS_clustering(d)

        elif (self.strategy == "DBSCAN"):

            dn = self.DBSCAN_clustering(d)

        elif (self.strategy == "HCA"):

            if self.metric not in ("cosine", "euclidean", "cityblock"):

                raise ValueError("The clustering metric should be cosine,"
                                 " euclidean or cityblock")

            try:

                dn = self.HCA_clustering(d, self.metric)

            except Exception:

                raise ValueError("The condensed distance matrix must "
                                 "contain only finite values")

        else:

            raise ValueError(
                "The Clustering function should be KMEANS, DBSCAN or HCA")

        silhouette = dn[0]

        clustd['train'] = dn[1]

        print("Clustering done -- CPU time: %s seconds" %
              (time.time() - start_time))

        return {'quality_metric': silhouette, 'result': clustd}
