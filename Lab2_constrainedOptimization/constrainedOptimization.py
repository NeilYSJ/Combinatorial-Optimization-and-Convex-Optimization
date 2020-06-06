# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: constrainedOptimization.py
@time: 2020/5/27 17:21
@desc:
'''

import numpy as np
from myClass import point, example, golden_search


def ALM(loss_function: example, start: point, lama: float, epsilon=1e-1, iteration_max=1000) -> list:
    """

    :param loss_function:
    :param start:
    :param lama:
    :param epsilon:
    :param iteration_max:
    :return:
    """
    points, M, k = [start], len(start), 0

    while True:
        # find the new point by cyclic coordinate method
        p = points[k]
        p_old = p
        while True:
            direction = [0] * M
            direction[np.mod(k, M)] = 1
            step = golden_search(loss_function, lama, p, direction)
            p = p + point(direction[0] * step, direction[1] * step)
            points.append(p)
            k += 1
            if k > iteration_max or (points[k] - points[k - 1]).L2() < epsilon: break
        # update the lama
        lama = lama + loss_function.rho * loss_function.subject_to(p)
        # if meet the termination condition then break
        if k > iteration_max or (p - p_old).L2() < epsilon: break

    return points


def ADMM(loss_function: example, start: point, lama: float, epsilon=1e-1, iteration_max=1000) -> list:
    """
    Alternating Direction Method of Multipliers
    :param loss_function:
    :param start:
    :param lama:
    :param epsilon:
    :param iteration_max:
    :return:
    """
    points, M, k = [start], len(start), 0

    while True:
        # update the point
        p = points[k]
        direction = [1, 0]
        step = golden_search(loss_function, lama, p, direction)
        p = p + point(direction[0] * step, direction[1] * step)
        direction = [0, 1]
        step = golden_search(loss_function, lama, p, direction)
        p = p + point(direction[0] * step, direction[1] * step)
        points.append(p)
        k += 1
        # update the lama
        lama = lama + loss_function.rho * loss_function.subject_to(p)
        # if meet the termination condition then break
        if k > iteration_max or (points[k] - points[k - 1]).L2() < epsilon: break

    return points
