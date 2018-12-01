#-*- coding : utf-8 -*-

import numpy as np

import networkx as nx          # graph in ISOMAP
import os

from sklearn.metrics import pairwise_distances
from sklearn.utils.graph import graph_shortest_path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA as StPCA


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
        
        vectors = vectors[index, :]
        
        self.A = vectors[0 : self.k, :]     # projection matrix
        
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
        dist_mat = pairwise_distances(X)                # distance matrix
        index_mat = np.argsort(dist_mat, axis = 1)      # sort
        
        while True:
            neighbors = index_mat[:, 1 : self.t + 1]  # neighbors_index for every point

            # construct graph
            gra = nx.Graph()                        # undirected graph
            nodes_list = list(range(X.shape[0]))    # nodes : 0, 1, 2, ..., n
            edges_list = [(i, j, {'weight' : dist_mat[i][j]}) for i in range(X.shape[0]) for j in neighbors[i]]   # edges with weight

            gra.add_edges_from(edges_list)          # construct graph
            
            if nx.is_connected(gra):
                break
            else:
                print("Neighobors = {}, not connected!".format(self.t))
                self.t *= 2
                self.t = min(self.t, X.shape[0] - 1)
        
        # search shortest path
        print("ISOMAP: Construct graph done! Search shortest path begin...")
        map_dist_mat = graph_shortest_path(nx.to_numpy_matrix(gra))
        print(map_dist_mat.max(), map_dist_mat.min())
        print("ISOMAP: Search shortest path done!")
        
        # distance_matrix to inner_product matrix
        mat_J = np.diag(np.ones(X.shape[0])) - 1.0/X.shape[0]
        inner_prod_mat = -0.5 * np.dot(np.dot(mat_J, map_dist_mat ** 2), mat_J)
        
        # eigen vectors
        eigens, vectors = np.linalg.eig(inner_prod_mat)    # eigen value decomposition
        eigens = np.abs(eigens)
        
        index = np.argsort(eigens)[::-1]    # sort descending
        
        eigens = eigens[index]
        vectors = vectors[index, :]
        
        low_x = np.dot(vectors[0 : self.k, :].transpose(), np.diag(np.sqrt(eigens[0 : self.k])))     # low dimensional x
        return low_x
        
        '''
        # svd
        U, S, V = np.linalg.svd(inner_prod_mat)    # sigular value decomposition
        
        U = U[:, 0 : self.k]
        S = S[0 : self.k]
        
        low_x = np.dot(U, np.diag(np.sqrt(S)))       # US^{1/2}
        return low_x
        '''

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

data_dir = "two datasets"
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
    stpca = StPCA(n_components = k)
    stpca.fit(train_x)
    low_train_x = stpca.transform(train_x)
    low_test_x = stpca.transform(test_x)
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(low_train_x, train_y)
    pre = knn.predict(low_test_x)
    accu = np.sum(pre == test_y) / len(test_y)
    print("#####################")
    print("Dataset = {}, sklearn.pca, low dimension k = {}, accuracy = {}...".format(name, k, accu))
    print("#####################")

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
