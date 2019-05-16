import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.decomposition import PCA


Cov = np.array([[2.9, -2.2], [-2.2, 6.5]])
print(Cov)
X = np.random.multivariate_normal([1, 2], Cov, size=200)
print(X.shape)
np.set_printoptions(4, suppress=True)

pca = PCA()
X_pca = pca.fit_transform(X)
print(pca.components_)
print(pca.mean_)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.axis('equal')
print(np.cov(X_pca, rowvar=False))
plt.show()

