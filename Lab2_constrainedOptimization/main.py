# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: main.py
@time: 2020/5/25 19:38
@desc:
'''


from myClass import point, example, drawResult
from constrainedOptimization import ALM, ADMM

if __name__ == '__main__':
    # python 3.6
    epsilon = 1e-2
    loss_function, start = example(rho=1.0), point(-2, -2)

    points = ALM(loss_function, start, lama=0, epsilon=epsilon)
    drawResult(loss_function, points, 'ALM', epsilon)

    # Alternating Direction Method of Multipliers
    points = ADMM(loss_function, start, lama=0, epsilon=epsilon)
    drawResult(loss_function, points, 'ADMM', epsilon)
