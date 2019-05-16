#!usr/bin/env python
#coding: utf-8

import math

def Tm(H, sigma, fai):
    R = 8.314
    return H/(50- R*math.log(sigma, math.e) + R*math.log(fai, math.e))


if __name__ == '__main__':
    print(Tm(28700, 4, 1480.3))



