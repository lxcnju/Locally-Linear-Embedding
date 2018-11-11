#-*- coding : utf-8 -*-
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

class ManifoldData():
    def __init__(self):
        pass

    def spiral3D(self, a = 1.0, b = 0.5, begin_circle = 0.0, end_circle = 1.5, num_points = 20000):
        theta = np.random.uniform(low = 2 * begin_circle * np.pi, high = 2 * end_circle * np.pi, size = (num_points, ))
        radius = a + b * theta
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)
        zs = np.random.uniform(low = 0.0, high = 10.0, size = (num_points,))
        return [ys, zs, xs]

    def plot_3d_data(self, data):
        fig = plt.figure()
        ax = Axes3D(fig)
        xs = data[0]
        ys = data[1]
        zs = data[2]
        colors = [xs[i] + ys[i] for i in range(len(xs))]
        ax.scatter(xs, ys, zs, s = 1, c = colors, cmap = plt.get_cmap("Set1"))
        plt.show()


ma = ManifoldData()
data1 = ma.spiral3D(begin_circle = 0.0, end_circle = 0.5, num_points = 5000)
data2 = ma.spiral3D(begin_circle = 0.5, end_circle = 1.0, num_points = 5000)
data3 = ma.spiral3D(begin_circle = 1.0, end_circle = 1.5, num_points = 5000)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data1[0], data1[1], data1[2], s = 1, c = "b", marker = "1")
ax.scatter(data2[0], data2[1], data2[2], s = 1, c = "y", marker = "1")
ax.scatter(data3[0], data3[1], data3[2], s = 1, c = "r", marker = "1")
plt.show()



data1 = np.array(data1).transpose()
data2 = np.array(data2).transpose()
data3 = np.array(data3).transpose()

data = np.vstack((data1, data2, data3))
print(data.shape)

pca = PCA(n_components = 3)
data_pca = pca.fit_transform(data)
components_ = pca.components_

print(components_)

data1 = data_pca[0:5000]
data2 = data_pca[5000:10000]
data3 = data_pca[10000:15000]


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], s = 1, c = "b", marker = "1")
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], s = 1, c = "y", marker = "1")
ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], s = 1, c = "r", marker = "1")
plt.show()


plt.figure()
plt.scatter(data1[:, 0], data1[:, 1], s = 1, c = "b", marker = "1")
plt.scatter(data2[:, 0], data2[:, 1], s = 1, c = "y", marker = "1")
plt.scatter(data3[:, 0], data3[:, 1], s = 1, c = "r", marker = "1")
plt.show()


plt.figure()
plt.scatter(data1[:, 1], data1[:, 2], s = 1, c = "b", marker = "1")
plt.scatter(data2[:, 1], data2[:, 2], s = 1, c = "y", marker = "1")
plt.scatter(data3[:, 1], data3[:, 2], s = 1, c = "r", marker = "1")
plt.show()

plt.figure()
plt.scatter(data1[:, 2], data1[:, 0], s = 1, c = "b", marker = "1")
plt.scatter(data2[:, 2], data2[:, 0], s = 1, c = "y", marker = "1")
plt.scatter(data3[:, 2], data3[:, 0], s = 1, c = "r", marker = "1")
plt.show()


from sklearn.manifold import locally_linear_embedding

for neis in [5, 10, 20, 30, 40, 50]:
    (data_lle, _) = locally_linear_embedding(data, n_neighbors = neis, n_components = 2)


    data1 = data_lle[0:5000]
    data2 = data_lle[5000:10000]
    data3 = data_lle[10000:15000]

    plt.figure()
    plt.scatter(data1[:, 0], data1[:, 1], s = 1, c = "b", marker = "1")
    plt.scatter(data2[:, 0], data2[:, 1], s = 1, c = "y", marker = "1")
    plt.scatter(data3[:, 0], data3[:, 1], s = 1, c = "r", marker = "1")
    plt.show()






