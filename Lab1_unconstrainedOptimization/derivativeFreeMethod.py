# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: derivativeFreeMethod.py
@time: 2020/5/16 17:47
@desc:
'''

import math
import numpy as np
from myClass import point, rosenbrock
from oneDimensionalSearch import golden_search, fibonacci_search, dichotomous_search
from oneDimensionalSearch import armijo_goldstein_search,wolfe_powell_search


def cyclic_coordinate_method(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-1,
                             k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param method:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, M, k = [start], len(start), 0

    while True:
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        # find the new x
        direction = [0] * M
        direction[np.mod(k, M)] = 1
        if method == 'golden_search':
            step = golden_search(loss_function, x[k], direction)
        elif method == 'fibonacci_search':
            step = fibonacci_search(loss_function, x[k], direction)
        elif method == 'dichotomous_search':
            step = dichotomous_search(loss_function, x[k], direction)
        else:
            return x
        x.append(x[k] + point(direction[0] * step, direction[1] * step))
        k += 1

    return x

