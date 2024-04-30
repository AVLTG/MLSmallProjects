import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Dataset
data = np.array([
    [1.39863187, -0.08561006, 1],
    [-0.36988611, 0.15146395, 1],
    [1.36395505, 0.98122297, 1],
    [-0.68197752, -0.75731975, 0],
    [-1.0980994, 1.33683399, 0],
    [-0.61262388, -1.62659111, 0]
])

# Separate features and labels
X = data[:, :-1]
y = data[:, -1]

# Create the 3NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier on the data
knn.fit(X, y)

# Create a mesh grid for plotting decision boundary
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define color maps for plotting
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3NN decision boundaries")

plt.show()
