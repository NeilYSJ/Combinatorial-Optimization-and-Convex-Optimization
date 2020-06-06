# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: hyperGradientDescent.py
@time: 2020/5/22 15:48
@desc:
'''

import numpy as np
from myClass import point, rosenbrock


def plain_gradient_descent_HD(loss_function: rosenbrock, start: point, initial_step=0.1, beta=0.001, epsilon=10e-2,
                              k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param step:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k, step = [start], 0, initial_step
    direction_old = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    p = step * direction_old
    x.append(x[k] + point(p[0], p[1]))

    while True:
        k += 1
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break

        # find the new x
        direction_new = -gradient / np.linalg.norm(gradient)
        # update the step
        step = step + beta * np.dot(direction_new, direction_old)
        p = step * direction_new
        x.append(x[k] + point(p[0], p[1]))

        direction_old = direction_new

    return x


def Nesterov_momentum_HD(loss_function: rosenbrock, start: point, initial_step=0.1, rho=0.7, mu=0.2, beta=0.001,
                         epsilon=10e-2, k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param step:
    :param rho: the influence of historical gradients
    :param beta: ahead rate
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k, step = [start], 0, initial_step
    direction_old = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    p = step * direction_old
    x.append(x[k] + point(p[0], p[1]))

    while True:
        k += 1
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(loss_function.g(x[k])) < epsilon: break

        # find the new x
        # ahead p * beta
        gradient = loss_function.g(x[k] + point(p[0] * mu, p[1] * mu))
        direction_new = -gradient / np.linalg.norm(gradient)
        # update the step
        step = step + beta * np.dot(direction_new, direction_old)
        # add the historical p
        p = rho * p + step * direction_new
        x.append(x[k] + point(p[0], p[1]))

        direction_old = direction_new

    return x


def Adam_HD(loss_function: rosenbrock, start: point, initial_step=0.1, rho0=0.9, rho1=0.99, beta=10e-7, epsilon=10e-2,
            k_max=10000) -> list:
    """
    Adaptive momentum
    :param loss_function:
    :param start:
    :param initial_step:
    :param rho0:
    :param rho1:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k, step = [start], 0, initial_step
    delta = np.array([10e-7] * len(start))

    r = np.zeros(len(start))
    direction = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    uk_old = direction
    r = rho1 * r + (1 - rho1) * direction ** 2
    p = step * uk_old
    x.append(x[k] + point(p[0], p[1]))

    while True:
        k += 1
        gradient = loss_function.g(x[k])
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        gradient = -gradient / np.linalg.norm(gradient)
        # add the influence rate of historical to direction
        direction = rho0 * direction + (1 - rho0) * gradient
        uk_new = direction / (r + delta) ** 0.5
        # find the new x
        # add the influence rate of historical to r
        r = rho1 * r + (1 - rho1) * gradient ** 2
        # update the step
        step = step + beta * np.dot(uk_new, uk_old)
        p = step * uk_new
        x.append(x[k] + point(p[0], p[1]))

        uk_old = uk_new

    return x
