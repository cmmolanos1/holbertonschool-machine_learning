#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Get an object of plasma colormap
color_map = plt.get_cmap('plasma')

fig = plt.figure()
ax = Axes3D(fig)

# Slicing pca_data
x = pca_data.copy()[:, 0]
y = pca_data.copy()[:, 1]
z = pca_data.copy()[:, 2]

for i in range(3):
    # Divide by axis by labels, using conditional slicing
    xs = x[labels == i]
    ys = y[labels == i]
    zs = z[labels == i]
    colors = color_map(i / 2)

    ax.scatter(xs, ys, zs, color=colors, marker="o", s=30)

ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

plt.title("PCA of Iris Dataset")
fig.set_tight_layout(False)
plt.show()
