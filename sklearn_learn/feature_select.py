from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
# X = iris.data
# Y = iris.target
# print(X.shape)
from sklearn.preprocessing import StandardScaler
# print(X)
# print(StandardScaler().fit_transform(X))

from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
test = SelectKBest(lambda X, Y: np.array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
print(test)


from sklearn.decomposition import PCA
# test = PCA(n_components=2).fit_transform(iris.data)

# from sklearn.linear_
# LDA(n_components=2).fit_transform(iris.data, iris.target)
# print(test.shape)