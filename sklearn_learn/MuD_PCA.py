import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.decomposition import PCA

Cov = np.array([[2.9, -2.2], [-2.2, 6.5]])
X = np.random.multivariate_normal([1, 2], Cov, size=200)
np.set_printoptions(4, suppress=True)
np.random.seed(1)
X_HD = np.dot(X, np.random.uniform(0.2, 3, (2, 4))*(np.random.randint(0, 2, (2, 4))*2-1))
print(X_HD[:10])
plt.figure(figsize=(8, 8))
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.scatter(X_HD[:, i], X_HD[:, j])
        plt.axis('equal')
        plt.gca().set_aspect('equal')
plt.show()
