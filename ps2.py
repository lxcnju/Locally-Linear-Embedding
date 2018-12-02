#-*- coding : utf-8 -*-

import numpy as np

import networkx as nx          # graph in ISOMAP
import os

from sklearn.metrics import pairwise_distances
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph


class PCA():
    def __init__(self, k):
        ''' k : low dimension
        '''
        self.k = k
    
    def fit(self, X):
        ''' X : shape = (n, d)
        '''
        X = X.copy()                        # copy
        
        mu = np.mean(X, axis = 0)           # zero center
        X = X - mu.reshape(1, -1)
        
        covX = np.dot(X.transpose(), X)     # covariance matrix
        
        eigens, vectors = np.linalg.eig(covX)    # eigen value decomposition
        
        index = np.argsort(eigens)[::-1]    # sort descending
        
        vectors = vectors[:, index]
        
        self.A = vectors[:, 0 : self.k].transpose()     # projection matrix
        
    def transform(self, X):
        ''' X : shape = (n, d)
        return : shape = (n, k)
        '''
        return np.dot(X, self.A.transpose())

class SVD():
    def __init__(self, k):
        ''' k : low dimension
        '''
        self.k = k
    
    def fit(self, X):
        ''' X : shape = (n, d)
        '''
        X = X.copy()                        # copy
        
        mu = np.mean(X, axis = 0)           # zero center
        X = X - mu.reshape(1, -1)
        
        U, S, V = np.linalg.svd(X)                 # sigular value decomposition
        
        self.A = V[0 : self.k, :]           # projection matrix
        
    def transform(self, X):
        ''' X : shape = (n, d)
        return : shape = (n, k)
        '''
        return np.dot(X, self.A.transpose())


class ISOMAP():
    def __init__(self, k, t):
        ''' k : low dimension
            t : number of nearest neighbors
        return : shape = (n, k)
        '''
        self.k = k
        self.t = t
        
    def fit_transform(self, X):
        ''' X : shape = (n, d)
        return : shape = (n, k)
        '''
        # find nearest neighbors
        neis = NearestNeighbors(n_neighbors = self.t)
        neis.fit(X)

        # construct neighbor-graph
        graph = kneighbors_graph(neis, self.t, mode = 'distance')

        # shortest path
        map_dist_mat = graph_shortest_path(graph, directed = False)

        # distance_matrix to inner_product matrix
        mat_J = np.diag(np.ones(X.shape[0])) - 1.0/X.shape[0]
        inner_prod_mat = -0.5 * np.dot(np.dot(mat_J, map_dist_mat ** 2), mat_J)
        
        # eigen vectors
        eigens, vectors = np.linalg.eigh(inner_prod_mat)    # eigen value decomposition
        
        index = np.argsort(eigens)[::-1][0 : self.k]        # sort descending
        
        eigens = eigens[index]
        vectors = vectors[:, index]
        
        low_x = np.dot(vectors, np.diag(np.sqrt(eigens)))            # low dimensional x
        #low_x = vectors
        return low_x
        

def read_data(fpath):
    xy = np.loadtxt(fpath, delimiter = ",")
    x = xy[:, 0:-1]
    y = xy[:, -1]
    return x, y
    
def read_datasets(data_dir, name):
    train_fpath = os.path.join(data_dir, "{}-train.txt".format(name))
    test_fpath = os.path.join(data_dir, "{}-test.txt".format(name))
    train_x, train_y = read_data(train_fpath)
    test_x, test_y = read_data(test_fpath)
    return train_x, train_y, test_x, test_y

data_dir = "./"
#name = "sonar"
name = "splice"
train_x, train_y, test_x, test_y = read_datasets(data_dir, name)

print("#####################")
print("Datasets = {}".format(name))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print("#####################")

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(train_x, train_y)
pre = knn.predict(test_x)
accu = np.sum(pre == test_y) / len(test_y)
print("#####################")
print("Dataset = {}, knn, accuracy = {}...".format(name, accu))
print("#####################")


for k in [10, 20, 30]:
    for method in ["PCA", "SVD", "ISOMAP"]:
        if method == "PCA":
            pca = PCA(k = k)
            pca.fit(train_x)

            low_train_x = pca.transform(train_x)
            low_test_x = pca.transform(test_x)
        elif method == "SVD":
            svd = SVD(k = k)
            svd.fit(train_x)
            
            low_train_x = svd.transform(train_x)
            low_test_x = svd.transform(test_x)
        elif method == "ISOMAP":
            isomap = ISOMAP(k = k, t = 30)
            
            X = np.vstack((train_x, test_x))
            
            low_X = isomap.fit_transform(X)
            
            low_train_x = low_X[0 : len(train_x), :]
            low_test_x = low_X[len(train_x) : ]
            

        knn = KNeighborsClassifier(n_neighbors = 1)
        knn.fit(low_train_x, train_y)

        pre = knn.predict(low_test_x)

        accu = np.sum(pre == test_y) / len(test_y)

        print("#####################")
        print("Dataset = {}, DimensionReduction method = {}, low dimension k = {}, accuracy = {}...".format(name, method, k, accu))
        print("#####################")
