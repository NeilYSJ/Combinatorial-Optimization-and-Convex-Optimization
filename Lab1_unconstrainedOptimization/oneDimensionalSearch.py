# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: oneDimensionalSearch.py
@time: 2020/5/19 14:05
@desc:
'''

import math
import random
import numpy as np
from myClass import point, rosenbrock


# Accurate One-dimensional Search

def advance_retreat_method(loss_function: rosenbrock, start: point, direction: list, step=0, delta=0.1) -> tuple:
    """
    find the initial section of step
    :param loss_function:
    :param start:
    :param direction:
    :param step:
    :param delta:
    :return:
    """
    alpha0, point0 = step, start

    alpha1 = alpha0 + delta
    point1 = point0 + point(direction[0] * delta, direction[1] * delta)
    if loss_function.f(point0) < loss_function.f(point1):
        while True:
            delta *= 2
            alpha2 = alpha0 - delta
            point2 = point0 - point(direction[0] * delta, direction[1] * delta)
            if loss_function.f(point2) < loss_function.f(point0):
                alpha1, alpha0 = alpha0, alpha2
                point1, point0 = point0, point2
            else:
                return alpha2, alpha1
    else:
        while True:
            delta *= 2
            alpha2 = alpha1 + delta
            point2 = point1 + point(direction[0] * delta, direction[1] * delta)
            if loss_function.f(point2) < loss_function.f(point1):
                alpha0, alpha1 = alpha1, alpha2
                point0, point1 = point1, point2
            else:
                return alpha0, alpha2


def golden_search(loss_function: rosenbrock, start: point, direction: list, epsilon=0.1) -> float:
    """
    derivative-free to search the longest step
    :param loss_function:
    :param start:
    :param direction:
    :param epsilon:
    :return:
    """
    a, b = advance_retreat_method(loss_function, start, direction)

    # find the minimum
    golden_num = (math.sqrt(5) - 1) / 2
    p, q = a + (1 - golden_num) * (b - a), a + golden_num * (b - a)
    while abs(a - b) > epsilon:
        f_p = loss_function.f(start + point(direction[0] * p, direction[1] * p))
        f_q = loss_function.f(start + point(direction[0] * q, direction[1] * q))
        if f_p < f_q:
            b, q = q, p
            p = a + (1 - golden_num) * (b - a)
        else:
            a, p = p, q
            q = a + golden_num * (b - a)

    return (a + b) / 2


def fibonacci_search(loss_function: rosenbrock, start: point, direction: list, epsilon=1) -> float:
    """
    derivative-free to search the longest step
    :param loss_function:
    :param start:
    :param direction:
    :param epsilon:
    :return:
    """
    a, b = advance_retreat_method(loss_function, start, direction)

    #  build the Fibonacci series
    F, d = [1.0, 2.0], (b - a) / epsilon
    while F[-1] < d: F.append(F[-1] + F[-2])

    # find the minimum
    N = len(F) - 1
    p, q = a + (1 - F[N - 1] / F[N]) * (b - a), a + F[N - 1] / F[N] * (b - a)
    while abs(a - b) > epsilon and N > 0:
        N = N - 1
        f_p = loss_function.f(start + point(direction[0] * p, direction[1] * p))
        f_q = loss_function.f(start + point(direction[0] * q, direction[1] * q))
        if f_p < f_q:
            b, q = q, p
            p = a + (1 - F[N - 1] / F[N]) * (b - a)
        else:
            a, p = p, q
            q = a + F[N - 1] / F[N] * (b - a)

    return (a + b) / 2


def dichotomous_search(loss_function: rosenbrock, start: point, direction: list, epsilon=0.1) -> float:
    """
    derivative-free to search the longest step
    :param loss_function:
    :param start:
    :param direction:
    :param epsilon:
    :return:
    """
    a, b = advance_retreat_method(loss_function, start, direction)

    # find the minimum
    e = epsilon / 3
    p, q = (a + b) / 2 - e, (a + b) / 2 + e
    while abs(a - b) > epsilon:
        f_p = loss_function.f(start + point(direction[0] * p, direction[1] * p))
        f_q = loss_function.f(start + point(direction[0] * q, direction[1] * q))
        if f_p < f_q:
            b = q
        else:
            a = p
        p, q = (a + b) / 2 - e, (a + b) / 2 + e

    return (a + b) / 2


# Inaccurate One-dimensional Search

def armijo_goldstein_search(loss_function: rosenbrock, start: point, direction: list, rho=0.1) -> float:
    """

    :param loss_function:
    :param start:
    :param direction:
    :param rho: meet condition 0<rho<0.5
    :return:
    """
    a, b = 0, 100
    alpha = b * random.uniform(0.5, 1)
    # find the alpha
    f1 = loss_function.f(start)
    gradient = loss_function.g(start)
    gradient_f1 = np.dot(gradient.T, np.array(direction))

    while True:
        f2 = loss_function.f(start + point(direction[0] * alpha, direction[1] * alpha))
        # print(f2 - f1, alpha)
        # print(rho * alpha * gradient_f1, (1 - rho) * alpha * gradient_f1)
        # the armijo goldstein rule
        # if alpha < 1: return 0.1
        if f2 - f1 <= rho * alpha * gradient_f1:
            if f2 - f1 >= (1 - rho) * alpha * gradient_f1:
                print(alpha)
                return alpha
            else:
                a, b = alpha, b
                if b < alpha:
                    alpha = (a + b) / 2
                else:
                    alpha = 2 * alpha
        else:
            a, b = a, alpha
            alpha = (a + b) / 2


def wolfe_powell_search(loss_function: rosenbrock, start: point, direction: list, rho=0.1) -> float:
    pass
