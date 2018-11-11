#-*- coding:utf-8 -*-


from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
from lle import LLE

mnist = input_data.read_data_sets("/home/lxcnju/workspace/datasets/mnist/")

num = 1000


X = mnist.train.images[0 : num]
Y = mnist.train.labels[0 : num]
uni_Y = np.unique(Y)

print(X.shape)
print(Y.shape)
print(uni_Y)


lle = LLE(k_neighbors = 10, low_dims = 100)
low_x = lle.fit_transform(X)

print("Done!")

