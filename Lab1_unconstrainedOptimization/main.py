# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: main.py
@time: 2020/5/16 16:25
@desc:
'''

import time
import pandas as pd
import numpy as np

from myClass import point, rosenbrock, drawResult, drawResult2, drawGifResult
from derivativeFreeMethod import cyclic_coordinate_method
from firstDerivativeMethod import steepest_descent, conjugate_gradient
from firstDerivativeMethod import plain_gradient_descent, Momentum, Nesterov_momentum
from firstDerivativeMethod import Adagrad, RMSprop, Adadelta, Adam
from secondDerivativeMethod import Newton_method, DFP, BFGS
from hyperGradientDescent import plain_gradient_descent_HD, Nesterov_momentum_HD, Adam_HD

from myClass import lassoregression, drawTheta
from lassoRegression import coordinate_descent, least_angle_regression


def mission_1():
    a, b, step = 1, 10, 0.1
    loss_function, start = rosenbrock(a, b), point(5, -10)

    ###### derivative-free method
    # step search method
    epsilon = 10e-1
    method = ['golden_search', 'fibonacci_search', 'dichotomous_search']
    # """
    for m in method:
        points = cyclic_coordinate_method(loss_function, start, method=m, epsilon=epsilon)
        #drawResult(loss_function, points, start, 'cyclic_coordinate_method by %s' % m, epsilon)
        drawResult2(loss_function, points, start, 'cyclic_coordinate_method by %s' % m, epsilon)
    # """

    ###### first derivative method
    # rho denotes the influence rate of historical p on the new p, mu denotes the ahead rate in Momentum
    # rho0 denotes the influence rate of historical direction on the new direction in Adam
    # rho1 denotes the influence rate of historical r on the new r in Adam
    epsilon = 10e-1
    rho, mu = 0.8, 0.2
    rho0, rho1 = 0.9, 0.99
    # """
    points = steepest_descent(loss_function, start, method=method[0], epsilon=epsilon)
    # drawResult(loss_function, points, start, 'steepest_descent by %s' % method[0], epsilon)
    drawResult2(loss_function, points, start, 'steepest_descent by %s' % method[0], epsilon)

    points = conjugate_gradient(loss_function, start, method=method[0], epsilon=epsilon)
    # drawResult(loss_function, points, start, 'conjugate_gradient by %s' % method[0], epsilon)
    drawResult2(loss_function, points, start, 'conjugate_gradient by %s' % method[0], epsilon)
    # """

    # """
    points = plain_gradient_descent(loss_function, start, step=step, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'plain_gradient_descent', epsilon, otherlabel=',step=%.2g' % step)
    drawResult2(loss_function, points, start, 'plain_gradient_descent', epsilon, otherlabel=',step=%.2g' % step)

    points = Momentum(loss_function, start, step=step, rho=rho, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Momentum', epsilon, otherlabel=',step=%.2g,rho=%.2g' % (step, rho))
    drawResult2(loss_function, points, start, 'Momentum', epsilon, otherlabel=',step=%.2g,rho=%.2g' % (step, rho))

    points = Nesterov_momentum(loss_function, start, step=step, rho=rho, mu=mu, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Nesterov_momentum', epsilon,
    #            otherlabel=',step=%.2g,rho=%.2g,mu=%.2g' % (step, rho, mu))
    drawResult2(loss_function, points, start, 'Nesterov_momentum', epsilon,
                otherlabel=',step=%.2g,rho=%.2g,mu=%.2g' % (step, rho, mu))
    # """

    # """
    points = Adagrad(loss_function, start, initial_step=step, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Adagrad', epsilon, otherlabel=',initial_step=%.2g' % step)
    drawResult2(loss_function, points, start, 'Adagrad', epsilon, otherlabel=',initial_step=%.2g' % step)

    points = RMSprop(loss_function, start, initial_step=step, rho=rho1, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'RMSProp', epsilon,
    #            otherlabel=',initial_step=%.2g,rho=%.2g' % (step, rho1))
    drawResult2(loss_function, points, start, 'RMSProp', epsilon,
                otherlabel=',initial_step=%.2g,rho=%.2g' % (step, rho1))

    points = Adadelta(loss_function, start, rho=rho1, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Adadelta', epsilon, otherlabel=',rho=%.2g' % rho1)
    drawResult2(loss_function, points, start, 'Adadelta', epsilon, otherlabel=',rho=%.2g' % rho1)

    points = Adam(loss_function, start, initial_step=step, rho0=rho0, rho1=rho1, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Adam', epsilon,
    #            otherlabel=',initial_step=%.2g,rho0=%.2g,rho1=%.2g' % (step, rho0, rho1))
    drawResult2(loss_function, points, start, 'Adam', epsilon,
                otherlabel=',initial_step=%.2g,rho0=%.2g,rho1=%.2g' % (step, rho0, rho1))
    # """

    ###### hyper gradient descent
    # beta denotes the influence rate of historical direction on the new step in hyperGradientDescent
    # """
    beta0, beta1 = 0.01, 10e-7

    points = plain_gradient_descent_HD(loss_function, start, initial_step=step, beta=beta0, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'plain_gradient_descent_HD', epsilon,
    #           otherlabel=',initial_step=%.2g,beta=%.2g' % (step, beta0))
    drawResult2(loss_function, points, start, 'plain_gradient_descent_HD', epsilon,
                otherlabel=',initial_step=%.2g,beta=%.2g' % (step, beta0))

    points = Nesterov_momentum_HD(loss_function, start, initial_step=step, rho=rho, mu=mu, beta=beta0, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Nesterov_momentum_HD', epsilon,
    #            otherlabel=',initial_step=%.2g,rho=%.2g,mu=%.2g,beta=%.2g' % (step, rho, mu, beta0))
    drawResult2(loss_function, points, start, 'Nesterov_momentum_HD', epsilon,
                otherlabel=',initial_step=%.2g,rho=%.2g,mu=%.2g,beta=%.2g' % (step, rho, mu, beta0))

    points = Adam_HD(loss_function, start, initial_step=step, rho0=rho0, rho1=rho1, beta=beta1, epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Adam_HD', epsilon,
    #            otherlabel=',initial_step=%.2g,rho0=%.2g,rho1=%.2g,beta=%.2g' % (step, rho0, rho1, beta1))
    drawResult2(loss_function, points, start, 'Adam_HD', epsilon,
                otherlabel=',initial_step=%.2g,rho0=%.2g,rho1=%.2g,beta=%.2g' % (step, rho0, rho1, beta1))
    # """

    ###### second derivative method
    # """
    epsilon = 10e-2

    points = Newton_method(loss_function, start, method=method[0], epsilon=epsilon)
    # drawResult(loss_function, points, start, 'Newton_method by %s' % method[0], epsilon)
    drawResult2(loss_function, points, start, 'Newton_method by %s' % method[0], epsilon)

    points = DFP(loss_function, start, method=method[0], epsilon=epsilon)
    # drawResult(loss_function, points, start, 'DFP by %s' % method[0], epsilon)
    drawResult2(loss_function, points, start, 'DFP by %s' % method[0], epsilon)

    points = BFGS(loss_function, start, method=method[0], epsilon=epsilon)
    # drawResult(loss_function, points, start, 'BFGS by %s' % method[0], epsilon)
    drawResult2(loss_function, points, start, 'BFGS by %s' % method[0], epsilon)
    # """


def mission_2():
    # data.columns denotes the number of features, data.index denotes the number of data
    data = pd.read_csv('sonar.all-data', header=None, prefix='V')
    data['V60'] = data.iloc[:, -1].apply(lambda v: 1.0 if v == 'M' else 0.0)
    # print(len(data.index),len(data.columns))
    # normalizing
    norm_data = (data - data.mean()) / data.std()
    # set the test data
    N = len(data.index) // 8
    X = data.iloc[N:, : -1].append(data.iloc[len(data.index) - N:, : -1])
    Y = data.iloc[N:, -1].append(data.iloc[len(data.index) - N:, -1])

    ###### coordinate_descent
    epsilon = 0.025
    loss_function = lassoregression(X=norm_data.values[N:len(data.index) - N, : -1],
                                    Y=norm_data.values[N:len(data.index) - N, -1],
                                    N=len(norm_data.index) - 2 * N)
    start = time.time()
    thetas = coordinate_descent(loss_function, epsilon=epsilon)
    end = time.time()
    # count the error of coordinate_descent
    theta = np.array(thetas[-1])
    error = np.linalg.norm(Y - X.dot(theta)) / len(Y.index)
    print("coordinate_descent\ttime:%f\terror:%f" % (end - start, error))
    # print(pd.value_counts(thetas[-1]))
    drawTheta('coordinate_descent (epsilon=%.3g)' % epsilon, thetas)

    ###### least_angle_regression
    start = time.time()
    thetas = least_angle_regression(X=norm_data.values[N:len(data.index) - N, : -1],
                                    Y=norm_data.values[N:len(data.index) - N, -1])
    end = time.time()
    # count the error of least_angle_regression
    theta = np.array(thetas[-1])
    error = np.linalg.norm(Y - X.dot(theta)) / len(Y.index)
    print("least_angle_regression\ttime:%f\terror:%f" % (end - start, error))
    # print(pd.value_counts(thetas[-1]))
    drawTheta('least_angle_regression', thetas)


if __name__ == '__main__':
    mission_1()
    # mission_2()
