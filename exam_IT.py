# !-*- coding: utf-8 -*-


# 用二分法实现开平方根
def square_root(c):
    i = 0
    max = c
    min = 0
    g = (max+min)/2.0
    while abs(g*g - c) > 0.000001:
        if g*g < c:
            min = g
        else:
            max = g
        g = (max + min)/2.0
        i += 1
        print('{0},{1:.6f}'.format(i, g))

square_root(10)

