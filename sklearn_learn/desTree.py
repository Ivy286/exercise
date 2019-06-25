import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num = int(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return train_test_split(X, y, test_size=0.25, random_state=1)


# 用决策树拟合
def test_DecisionTreeRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)
    print("Training score:%f" % (regr.score(X_train, y_train)))
    print("Test score:%f" % (regr.score(X_test, y_test)))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train, y_train, label="train sample", c='g')
    ax.scatter(X_test, y_test, label="test sample", c='r')
    ax.plot(X, Y, label="predict value", linewidth=2, alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = creat_data(100)
    print(X_train, X_test, y_train, y_test)
    test_DecisionTreeRegressor(X_train, X_test, y_train, y_test)
