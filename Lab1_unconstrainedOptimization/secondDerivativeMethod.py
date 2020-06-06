# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: secondDerivativeMethod.py
@time: 2020/5/16 17:45
@desc:
'''

import numpy as np
from myClass import point, rosenbrock
from oneDimensionalSearch import golden_search, fibonacci_search, dichotomous_search


def Newton_method(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-2, k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param step:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0

    while True:
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break

        # find the new x
        inverse = np.linalg.inv(loss_function.H(x[k]))
        direction = -np.matmul(inverse, gradient)
        if method == 'golden_search':
            step = golden_search(loss_function, x[k], direction)
        elif method == 'fibonacci_search':
            step = fibonacci_search(loss_function, x[k], direction)
        elif method == 'dichotomous_search':
            step = dichotomous_search(loss_function, x[k], direction)
        else:
            return x
        p = step * direction
        x.append(x[k] + point(p[0], p[1]))
        k += 1

    return x


def DFP(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-2, k_max=10000) -> list:
    """
    Davidon Fletcher Powell, quasi_newton_method
    :param loss_function:
    :param start:
    :param step:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    D = np.identity(len(start))  # Identity matrix

    while True:
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break

        # find the new x
        gradient = gradient / np.linalg.norm(gradient)
        direction = -np.matmul(D, gradient)
        if method == 'golden_search':
            step = golden_search(loss_function, x[k], direction)
        elif method == 'fibonacci_search':
            step = fibonacci_search(loss_function, x[k], direction)
        elif method == 'dichotomous_search':
            step = dichotomous_search(loss_function, x[k], direction)
        else:
            return x
        p = step * direction
        x.append(x[k] + point(p[0], p[1]))
        # update the D
        yk = np.mat(loss_function.g(x[k + 1]) / np.linalg.norm(loss_function.g(x[k + 1])) - gradient).T
        pk = np.mat(p).T
        Dk = np.mat(D)
        D = D + np.array((pk * pk.T) / (pk.T * yk) - (Dk * yk * yk.T * Dk) / (yk.T * Dk * yk))
        k += 1

    return x


def BFGS(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-2, k_max=10000) -> list:
    """
    Broyden Fletcher Goldfarb Shanno
    :param loss_function:
    :param start:
    :param method:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    B = np.identity(len(start))  # Identity matrix

    while True:
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break

        # find the new x
        gradient = gradient / np.linalg.norm(gradient)
        direction = -np.matmul(np.linalg.inv(B), gradient)
        if method == 'golden_search':
            step = golden_search(loss_function, x[k], direction)
        elif method == 'fibonacci_search':
            step = fibonacci_search(loss_function, x[k], direction)
        elif method == 'dichotomous_search':
            step = dichotomous_search(loss_function, x[k], direction)
        else:
            return x
        p = step * direction
        x.append(x[k] + point(p[0], p[1]))
        # update the B
        yk = np.mat(loss_function.g(x[k + 1]) / np.linalg.norm(loss_function.g(x[k + 1])) - gradient).T
        pk = np.mat(p).T
        Bk = np.mat(B)
        B = B + np.array((yk * yk.T) / (yk.T * pk) - (Bk * pk * pk.T * Bk) / (pk.T * Bk * pk))
        k += 1

    return x
