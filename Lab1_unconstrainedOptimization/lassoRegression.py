# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: lassoRegression.py
@time: 2020/5/23 14:42
@desc:
'''

# LASSO Regression: Least Absolute Shrinkage and Selection Operator Regression whose loss function is not differentiable
# CD/LARS: coordinate descent/least Angle Regression

import math
import pandas as pd
import numpy as np
import numpy.matlib
from myClass import lassoregression


def advance_retreat_method(loss_function: lassoregression, start: np.mat, direction: np.mat, step=0,
                           delta=0.1) -> tuple:
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
    point1 = point0 + direction * delta
    if loss_function.f(point0) < loss_function.f(point1):
        while True:
            delta *= 2
            alpha2 = alpha0 - delta
            point2 = point0 - direction * delta
            if loss_function.f(point2) < loss_function.f(point0):
                alpha1, alpha0 = alpha0, alpha2
                point1, point0 = point0, point2
            else:
                return alpha2, alpha1
    else:
        while True:
            delta *= 2
            alpha2 = alpha1 + delta
            point2 = point1 + direction * delta
            if loss_function.f(point2) < loss_function.f(point1):
                alpha0, alpha1 = alpha1, alpha2
                point0, point1 = point1, point2
            else:
                return alpha0, alpha2


def golden_search(loss_function: lassoregression, start: np.mat, direction: np.mat, epsilon=0.1) -> float:
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
        f_p = loss_function.f(start + direction * p)
        f_q = loss_function.f(start + direction * q)
        if f_p < f_q:
            b, q = q, p
            p = a + (1 - golden_num) * (b - a)
        else:
            a, p = p, q
            q = a + golden_num * (b - a)

    return (a + b) / 2


def coordinate_descent(loss_function: lassoregression, epsilon=10e-2, k_max=350) -> list:
    """

    :param loss_function:
    :param epsilon:
    :param k_max:
    :return:
    """
    theta = np.matlib.zeros((loss_function.X.shape[1], 1))
    M, k = theta.shape[0], 0

    temp = np.around(theta.T.tolist()[0], 3)
    theta_list = [temp]
    direction = np.matlib.zeros((M, 1))
    while True:
        # update the theta on each dimension
        for i in range(M):
            direction[np.mod(i - 1, M), 0] = 0
            direction[np.mod(i, M), 0] = 1
            step = golden_search(loss_function, theta, direction)
            theta = theta + direction * step
        k += 1
        temp = np.around(theta.T.tolist()[0], 3)
        theta_list.append(temp)
        # if meet the termination conditions then break
        if k > k_max or np.max(abs(theta_list[k] - theta_list[k - 1])) < epsilon:
            break

    return theta_list


def least_angle_regression(X: pd.DataFrame, Y: pd.DataFrame, k_max=350) -> list:
    """

    :param X:
    :param Y:
    :param k_max:
    :return:
    """
    # set the length of step
    step = 0.004

    theta = np.array([0.0] * X.shape[1])
    theta_list = [list(theta)]
    for k in range(k_max):
        # count the residual
        residual = Y - X.dot(theta)
        # count the correlation between features and residual
        corr = residual.dot(X)
        # find the feature which has the max correlation, update its theta
        idx = np.abs(corr).argmax()
        corrMax = corr[idx]
        theta[idx] += step * corrMax / abs(corrMax)
        # save the result
        theta_list.append(list(theta))

    return theta_list
