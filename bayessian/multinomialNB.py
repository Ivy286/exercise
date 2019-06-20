import numpy as np

from sklearn.naive_bayes import MultinomialNB

np.random.seed(0)
M = 20
N = 5
x = np.random.randint(2, size=(M, N))
print(x)
x = np.array(list(set([tuple(t) for t in x])))
print(x)
M = len(x)
y = np.arange(M)
print(y)
mnb = MultinomialNB(alpha=1)
mnb.fit(x, y)
y_hat = mnb.predict(x)
print('y_hat:', y_hat)
print('prediction precision: %.2f%%' % (100*np.mean(y_hat == y)))  # np.mean(x==y) 返回条件成立的占比
print('系统得分：', mnb.score(x, y))
