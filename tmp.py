import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# 直线方程函数
def f_1(x, A, B):
    return A * x + B


# 二次曲线方程
def f_2(x, A, B, C):
    return A * x * x + B * x + C


# 三次曲线方程
def f_3(x, A, B, C, D):
    return A * x * x * x + B * x * x + C * x + D


def plot_test():
    plt.figure()
    x0 = [2, 4, 5, 6, 8, 10]
    y0 = [0.324, 0.458, 0.477, 0.481, 0.382, -0.01]
    y1 = [0.169, 0.438, 0.322, 0.49, 0.186, 0.001]

    # 绘制散点
    plt.scatter(x0[:], y0[:], 25,  "red")
    plt.scatter(x0[:], y1[:], 25,  "green")

    # # 二次曲线拟合与绘制
    A21, B21, C21 = optimize.curve_fit(f_2, x0, y0)[0]
    A22, B22, C22 = optimize.curve_fit(f_2, x0, y1)[0]
    x2 = np.arange(0, 12, 0.5)
    y21 = A21 * x2 * x2 + B21 * x2 + C21
    y22 = A22 * x2 * x2 + B22 * x2 + C22
    plt.plot(x2, y21, "red", label='mannual model')
    plt.plot(x2, y22, 'g--', label='auto model')
    plt.vlines(4, 0, 1, linestyles='dotted')

    # # 三次曲线拟合与绘制
    # A3, B3, C3, D3 = optimize.curve_fit(f_3, x0, y0)[0]
    # x3 = np.arange(0, 12, 0.5)
    # y3 = A3 * x3 * x3 * x3 + B3 * x3 * x3 + C3 * x3 + D3
    # plt.plot(x3, y3, "purple")

    plt.title("")
    plt.xlabel('maxPath')
    plt.ylabel('$r^2$ score')
    plt.legend(loc=3)
    plt.show()
    return

plot_test()