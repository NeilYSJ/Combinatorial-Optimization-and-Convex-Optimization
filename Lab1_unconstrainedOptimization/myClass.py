# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: myClass.py
@time: 2020/5/16 18:09
@desc:
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class point:
    x, y = 0, 0

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y)

    def __len__(self):
        return 2

    def __str__(self):
        return str(self.x) + "," + str(self.y)


class rosenbrock:
    a, b = 1.0, 1.0

    def __init__(self, a, b):
        self.a, self.b = a * 1.0, b * 1.0

    def f(self, p: point):
        return (self.a - p.x) ** 2 + self.b * (p.y - p.x ** 2) ** 2

    def g(self, p: point):
        return np.array([-4 * self.b * p.x * (p.y - p.x ** 2) - 2 * (self.a - p.x), 2 * self.b * (p.y - p.x ** 2)])

    def H(self, p: point):
        return np.array(
            [[12 * self.b * p.x * p.x - 4 * self.b * p.y + 2, -4 * self.b * p.x], [-4 * self.b * p.x, 2 * self.b]])


class lassoregression:
    beta, N = 0.002, 0
    X, Y = 0, 0

    def __init__(self, X, Y, N):
        self.X, self.Y, self.N = np.mat(X), np.mat(Y).T, N

    def f(self, theta: np.array):
        return (((self.X * theta - self.Y).T * (self.X * theta - self.Y)) / (2 * self.N)
                + self.beta * np.linalg.norm(theta, ord=1))[0, 0]

def drawTheta(label: str, result: list):
    plt.plot(result)
    plt.grid(True)

    plt.title(label)
    plt.xlabel("step")
    plt.ylabel("theta")

    plt.show()


def drawResult(loss_function: rosenbrock, points: list, start: point, label: str, epsilon: float, otherlabel=''):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title(
        label + ' (a=' + str(loss_function.a) + ',b=' + str(
            loss_function.b) + otherlabel + ',epsilon=' + '%.e' % epsilon + ',iteration=' + str(len(points)) + ')')
    # draw the function
    X = np.arange(-5.5, 5.5 + 0.05, 0.05)
    Y = np.arange(-30, 30 + 0.05, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = loss_function.f(point(X, Y))
    ax.plot_surface(Y, X, Z, cmap='rainbow', alpha=0.55)
    plt.contour(Y, X, Z, 60, colors='black', alpha=0.55)
    # draw the line of points
    x, y, z = [], [], []
    for p in points:
        x.append(p.x)
        y.append(p.y)
        z.append(loss_function.f(p))
    ax.plot3D(y, x, z, 'r*-')
    # draw the start point
    ax.scatter(start.y, start.x, loss_function.f(start), color='black')
    ax.text(start.y, start.x, loss_function.f(start), 'start(' + str(start.x) + ',' + str(start.y) + ')')
    # draw the end point
    end = points[len(points) - 1]
    ax.scatter(end.y, end.x, loss_function.f(end), color='black')
    ax.text(end.y, end.x, loss_function.f(end),
            'end(' + '%.2f' % end.x + ',' + '%.2f' % end.y + ',' + '%.2f' % loss_function.f(end) + ')',
            verticalalignment='top')
    # draw the target point
    target = point(loss_function.a, loss_function.a ** 2)
    ax.scatter(target.y, target.x, loss_function.f(target), color='black')
    ax.text(target.y, target.x, loss_function.f(target),
            'target(' + str(target.x) + ',' + str(target.y) + ',' + str(loss_function.f(target)) + ')',
            verticalalignment='bottom')

    plt.show()


def drawGifResult(loss_function: rosenbrock, points: list, start: point, label: str, epsilon: float, otherlabel=''):
    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    # draw the function
    plt.title(
        label + ' (a=' + str(loss_function.a) + ',b=' + str(
            loss_function.b) + otherlabel + ',epsilon=' + '%.e' % epsilon + ',iteration=' + str(len(points)) + ')')
    X = np.arange(-5.5, 5.5 + 0.05, 0.05)
    Y = np.arange(-30, 30 + 0.05, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = loss_function.f(point(X, Y))
    ax.plot_surface(Y, X, Z, cmap='rainbow', alpha=0.55)
    plt.contour(Y, X, Z, 60, colors='black', alpha=0.55)
    # draw the start point
    ax.scatter(start.y, start.x, loss_function.f(start), color='black')
    ax.text(start.y, start.x, loss_function.f(start), 'start(' + str(start.x) + ',' + str(start.y) + ')')
    # draw the target point
    target = point(loss_function.a, loss_function.a ** 2)
    ax.scatter(target.y, target.x, loss_function.f(target), color='black')
    ax.text(target.y, target.x, loss_function.f(target),
            'target(' + str(target.x) + ',' + str(target.y) + ',' + str(loss_function.f(target)) + ')',
            verticalalignment='bottom')
    # draw the line of points
    k = 0
    end = points[len(points) - 1]
    x, y, z = [], [], []
    while True:
        if k == len(points):
            ax.plot3D(y, x, z, 'r*-')
            ax.scatter(end.y, end.x, loss_function.f(end), color='black')
            ax.text(end.y, end.x, loss_function.f(end),
                    'end(' + '%.2f' % end.x + ',' + '%.2f' % end.y + ',' + '%.2f' % loss_function.f(end) + ')',
                    verticalalignment='top')
            break
        x.append(points[k].x)
        y.append(points[k].y)
        z.append(loss_function.f(points[k]))
        ax.plot3D(y, x, z, 'r*-')
        k += 1
        plt.pause(0.1)

    plt.pause(1000)

def drawResult2(loss_function: rosenbrock, points: list, start: point, label: str, epsilon: float, otherlabel=''):
    plt.figure()
    plt.title(
        label + ' (a=' + str(loss_function.a) + ',b=' + str(
            loss_function.b) +',iteration=' + str(len(points)) + ')')
    # draw the function
    X = np.arange(-5.5, 5.5 + 0.05, 0.05)
    Y = np.arange(-30, 30 + 0.05, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = loss_function.f(point(X, Y))
    contour = plt.contour(Y, X, Z, colors='k')
    plt.clabel(contour, fontsize=8, colors='k')

    # draw the line of points
    x, y= [], []
    for p in points:
        x.append(p.x)
        y.append(p.y)
    plt.plot(y, x, 'r*-')
    # draw the start point
    plt.scatter(start.y, start.x,  color='black')
    plt.text(start.y, start.x, 'start(' + str(start.x) + ',' + str(start.y) + ')')
    # draw the end point
    end = points[len(points) - 1]
    plt.scatter(end.y, end.x, color='black')
    plt.text(end.y, end.x,
            'end(' + '%.2f' % end.x + ',' + '%.2f' % end.y + ',' + '%.2f' % loss_function.f(end) + ')',
            verticalalignment='top')
    # draw the target point
    target = point(loss_function.a, loss_function.a ** 2)
    plt.scatter(target.y, target.x,  color='black')
    plt.text(target.y, target.x,
            'target(' + str(target.x) + ',' + str(target.y) + ',' + str(loss_function.f(target)) + ')',
            verticalalignment='bottom')

    plt.show()