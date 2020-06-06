# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: myClass.py
@time: 2020/5/30 15:26
@desc:
'''

import math
import numpy as np
import matplotlib.pyplot as plt


class point:
    x, y = 0, 0

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y)

    def __len__(self):
        return 2

    def __str__(self):
        return str(self.x) + "," + str(self.y)

    def L2(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


class example:
    rho = 1.0

    def __init__(self, rho: float):
        self.rho = rho

    def F(self, p: point):
        return (p.x - 1) ** 2 + (p.y - 2) ** 2

    def subject_to(self, p: point):
        return 2 * p.x + 3 * p.y - 5

    def L(self, p: point, lam: float):
        return self.F(p) + lam * self.subject_to(p) + 0.5 * self.rho * (self.subject_to(p) ** 2)


def advance_retreat_method(loss_function: example, lama: float, start: point, direction: list, step=0,
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
    point1 = point0 + point(direction[0] * delta, direction[1] * delta)
    if loss_function.L(point0, lama) < loss_function.L(point1, lama):
        while True:
            delta *= 2
            alpha2 = alpha0 - delta
            point2 = point0 - point(direction[0] * delta, direction[1] * delta)
            if loss_function.L(point2, lama) < loss_function.L(point0, lama):
                alpha1, alpha0 = alpha0, alpha2
                point1, point0 = point0, point2
            else:
                return alpha2, alpha1
    else:
        while True:
            delta *= 2
            alpha2 = alpha1 + delta
            point2 = point1 + point(direction[0] * delta, direction[1] * delta)
            if loss_function.L(point2, lama) < loss_function.L(point1, lama):
                alpha0, alpha1 = alpha1, alpha2
                point0, point1 = point1, point2
            else:
                return alpha0, alpha2


def golden_search(loss_function: example, lama: float, start: point, direction: list, epsilon=0.1) -> float:
    """
    derivative-free to search the longest step
    :param loss_function:
    :param start:
    :param direction:
    :param epsilon:
    :return:
    """
    a, b = advance_retreat_method(loss_function, lama, start, direction)

    # find the minimum
    golden_num = (math.sqrt(5) - 1) / 2
    p, q = a + (1 - golden_num) * (b - a), a + golden_num * (b - a)
    while abs(a - b) > epsilon:
        f_p = loss_function.L(start + point(direction[0] * p, direction[1] * p), lama)
        f_q = loss_function.L(start + point(direction[0] * q, direction[1] * q), lama)
        if f_p < f_q:
            b, q = q, p
            p = a + (1 - golden_num) * (b - a)
        else:
            a, p = p, q
            q = a + golden_num * (b - a)

    return (a + b) / 2


def drawResult(loss_function: example, points: list, label: str, epsilon: float, other_label=''):
    plt.figure()
    plt.title(
        label + '(rho=' + str(loss_function.rho) + other_label + ',epsilon=' + '%.e' % epsilon + ',iteration=' + str(
            len(points)) + ')')

    # draw the function and condition
    X = np.arange(-2, 4.5 + 0.05, 0.05)
    Y = np.arange(-2, 3 + 0.05, 0.05)
    Y2 = (5 - 2 * X) / 3
    plt.plot(X, Y2, 'red')
    X, Y = np.meshgrid(X, Y)
    Z1 = loss_function.F(point(X, Y))
    contour2 = plt.contour(X, Y, Z1, colors='k')
    plt.clabel(contour2, fontsize=8, colors='k')

    # draw the result
    x, y = [], []
    for p in points:
        x.append(p.x)
        y.append(p.y)
    plt.plot(x, y, 'b*-')
    contour1 = plt.contour(X, Y, Z1, [loss_function.F(points[-1])], colors='blue')
    plt.clabel(contour1, inline=True, fontsize=8, colors='blue')

    # draw the start point
    plt.scatter(points[0].x, points[0].y, color='blue')
    plt.text(points[0].x, points[0].y, 'start(%.3g,%.3g,%.3g)' % (points[0].x, points[0].y, loss_function.F(points[0])),
             color='blue', verticalalignment='top')
    # draw the end point
    plt.scatter(points[-1].x, points[-1].y, color='blue')
    plt.text(points[-1].x, points[-1].y,
             'end(%.3g,%.3g,%.3g)' % (points[-1].x, points[-1].y, loss_function.F(points[-1])), color='blue',
             verticalalignment='bottom')
    # draw the target point
    target = point(0.53846, 1.30769)
    plt.scatter(target.x, target.y, color='red')
    plt.text(target.x, target.y,
             'target(%.3g,%.3g,%.3g)' % (target.x, target.y, loss_function.F(target)), color='red',
             verticalalignment='top', horizontalalignment='right')

    plt.show()
