#-×- coding : utf-8 -*-

# 实现Locally Linear Embedding


import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigs as sparse_eigs

class LLE():
    def __init__(self, k_neighbors, low_dims):
        '''
        init function
        @params k_neighbors : the number of neigbors
        @params low_dims : low dimension
        '''
        self.k_neighbors = k_neighbors
        self.low_dims = low_dims

    def fit_transform(self, X):
        '''
        transform X to low-dimension
        @params X : 2-d numpy.array (n_samples, high_dims)
        '''
        n_samples = X.shape[0]
        # calculate pair-wise distance
        dist_mat = pairwise_distances(X)
        # index of neighbors, not include self
        neighbors = np.argsort(dist_mat, axis = 1)[:, 1 : self.k_neighbors + 1]
        # neighbor combination matrix
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            mat_z = X[i] - X[neighbors[i]]
            mat_c = np.dot(mat_z, mat_z.transpose())
            w = np.linalg.solve(mat_c, np.ones(mat_c.shape[0]))
            W[i, neighbors[i]] = w / w.sum()
        # sparse matrix M
        I_W = np.eye(n_samples) - W
        M = np.dot(I_W.transpose(), I_W)
        # solve the d+1 lowest eigen values
        eigen_values, eigen_vectors = np.linalg.eig(M)
        index = np.argsort(eigen_values)[1 : self.low_dims + 1]
        selected_eig_values = eigen_values[index]
        selected_eig_vectors = eigen_vectors[index]

        self.eig_values = selected_eig_values
        self.low_X = selected_eig_vectors.transpose()
        print(self.low_X.shape)
        return self.low_X















