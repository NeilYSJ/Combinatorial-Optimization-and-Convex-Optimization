# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: firstDerivativeMethod.py
@time: 2020/5/16 17:45
@desc:
'''

import numpy as np
from myClass import point, rosenbrock
from oneDimensionalSearch import golden_search, fibonacci_search, dichotomous_search
from oneDimensionalSearch import armijo_goldstein_search, wolfe_powell_search


def steepest_descent(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-2,
                     k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param method:
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
        direction = -gradient / np.linalg.norm(gradient)
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


def conjugate_gradient(loss_function: rosenbrock, start: point, method='golden_search', epsilon=10e-2,
                       k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param method:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, direction, k = [start], -1 * loss_function.g(start) / np.linalg.norm(loss_function.g(start)), 0

    while True:
        # if meet the termination conditions then break
        gradient_old = loss_function.g(x[k]) / np.linalg.norm(loss_function.g(x[k]))
        if np.linalg.norm(loss_function.g(x[k])) < epsilon or k > k_max: break

        # find the new x
        if method == 'golden_search':
            step = golden_search(loss_function, x[k], direction)
        elif method == 'fibonacci_search':
            step = fibonacci_search(loss_function, x[k], direction)
        elif method == 'dichotomous_search':
            step = dichotomous_search(loss_function, x[k], direction)
        else:
            return x
        x.append(x[k] + point(direction[0] * step, direction[1] * step))

        # update the direction
        gradient_new = loss_function.g(x[k + 1]) / np.linalg.norm(loss_function.g(x[k + 1]))
        alpha = np.dot(gradient_new, gradient_new) / np.dot(gradient_old, gradient_old)
        direction = -gradient_new + alpha * direction
        k += 1

    return x


def plain_gradient_descent(loss_function: rosenbrock, start: point, step=0.1, epsilon=10e-2, k_max=10000) -> list:
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
        direction = -gradient / np.linalg.norm(gradient)
        p = step * direction
        x.append(x[k] + point(p[0], p[1]))
        k += 1

    return x


def Momentum(loss_function: rosenbrock, start: point, step=0.1, rho=0.7, epsilon=10e-2, k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param step:
    :param rho: the influence of historical gradients
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    direction = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    p = step * direction
    x.append(x[k] + point(p[0], p[1]))

    while True:
        k += 1
        # if meet the termination conditions then break
        gradient = loss_function.g(x[k])
        if k > k_max or np.linalg.norm(gradient) < epsilon: break

        # find the new x
        direction = -gradient / np.linalg.norm(gradient)
        # add the historical p
        p = rho * p + step * direction
        x.append(x[k] + point(p[0], p[1]))

    return x


def Nesterov_momentum(loss_function: rosenbrock, start: point, step=0.1, rho=0.7, mu=0.2, epsilon=10e-2,
                      k_max=10000) -> list:
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
    x, k = [start], 0
    direction = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    p = step * direction
    x.append(x[k] + point(p[0], p[1]))

    while True:
        k += 1
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(loss_function.g(x[k])) < epsilon: break

        # find the new x
        # ahead p * mu
        gradient = loss_function.g(x[k] + point(p[0] * mu, p[1] * mu))
        direction = -gradient / np.linalg.norm(gradient)
        # add the historical p
        p = rho * p + step * direction
        x.append(x[k] + point(p[0], p[1]))

    return x


def Adagrad(loss_function: rosenbrock, start: point, initial_step=0.1, epsilon=10e-2, k_max=10000) -> list:
    """
    Adaptive Gradient
    :param loss_function:
    :param start:
    :param initial_step:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    delte = np.array([10e-7] * len(start))

    r = np.zeros(len(start))
    while True:
        gradient = loss_function.g(x[k])
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        gradient = -gradient / np.linalg.norm(gradient)
        # find the new x
        r = r + gradient ** 2
        p = initial_step * gradient / (r + delte) ** 0.5
        x.append(x[k] + point(p[0], p[1]))
        k += 1

    return x


def RMSprop(loss_function: rosenbrock, start: point, initial_step=0.1, rho=0.99, epsilon=10e-2, k_max=10000) -> list:
    """
    root mean squared
    :param loss_function:
    :param start:
    :param initial_step:
    :param rho:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    delta = np.array([10e-7] * len(start))

    r = np.zeros(len(start))
    while True:
        gradient = loss_function.g(x[k])
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        gradient = -gradient / np.linalg.norm(gradient)
        # find the new x
        # add the influence rate of historical to r
        r = rho * r + (1 - rho) * gradient ** 2
        p = initial_step * gradient / (r + delta) ** 0.5
        x.append(x[k] + point(p[0], p[1]))
        k += 1

    return x


def Adadelta(loss_function: rosenbrock, start: point, rho=0.99, epsilon=10e-2, k_max=10000) -> list:
    """

    :param loss_function:
    :param start:
    :param rho:
    :param epsilon:
    :param k_max:
    :return:
    """
    x, k = [start], 0
    delta = np.array([10e-7] * len(start))

    step = np.zeros(len(start))
    r = np.zeros(len(start))
    while True:
        gradient = loss_function.g(x[k])
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        gradient = -gradient / np.linalg.norm(gradient)
        # find the new x
        # add the influence rate of historical to r
        r = rho * r + (1 - rho) * gradient ** 2
        p = gradient * ((step + delta) / (r + delta)) ** 0.5
        x.append(x[k] + point(p[0], p[1]))
        step = rho * step + (1 - rho) * p ** 2
        k += 1

    return x


def Adam(loss_function: rosenbrock, start: point, initial_step=0.1, rho0=0.9, rho1=0.99, epsilon=10e-2,
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
    x, k = [start], 0
    delta = np.array([10e-7] * len(start))

    r = np.zeros(len(start))
    direction = -loss_function.g(start) / np.linalg.norm(loss_function.g(start))
    while True:
        gradient = loss_function.g(x[k])
        # if meet the termination conditions then break
        if k > k_max or np.linalg.norm(gradient) < epsilon: break
        gradient = -gradient / np.linalg.norm(gradient)
        # add the influence rate of historical to direction
        direction = rho0 * direction + (1 - rho0) * gradient
        # find the new x
        # add the influence rate of historical to r
        r = rho1 * r + (1 - rho1) * gradient ** 2
        p = initial_step * direction / (r + delta) ** 0.5
        x.append(x[k] + point(p[0], p[1]))
        k += 1

    return x
