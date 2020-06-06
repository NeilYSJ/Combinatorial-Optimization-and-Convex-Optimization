# -*- coding: utf-8 -*-
'''
@author: Neil.YU
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: neil_yushengjian@foxmail.com
@software: PyCharm 2018.1.2
@file: PCA.py
@time: 2020/6/1 20:38
@desc:
'''

import os
import numpy as np
from scipy import misc
from glob import glob
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from matplotlib import cm
from numpy.linalg import norm, svd


def PCA_IALM(X, lam=0.01, tol=1e-7, maxIter=1000):
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lam
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[1]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lam / mu, 0) + np.minimum(Eraw + lam / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(0.05 * n), n])

        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        # print itr
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxIter):
            break
    print("iteration:%d" % (itr))
    return A, E


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / 255


def make_video(alg, cache_path='./ShoppingMall/result'):
    name = alg
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    mat = loadmat("./ShoppingMall/IALM_background_subtraction.mat")
    org = X.reshape(d1, d2, X.shape[1]) * 255.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    usable = [x for x in sorted(mat.keys()) if "_" not in x][0]
    sz = min(org.shape[2], mat[usable].shape[2])
    for i in range(sz):
        ax.cla()
        ax.axis("off")
        ax.imshow(np.hstack([mat[x][:, :, i] for x in sorted(mat.keys()) if "_" not in x] + \
                            [org[:, :, i]]), cm.gray)
        fname_ = '%s/%s_result%03d.png' % (cache_path, name, i)
        fig.tight_layout()
        fig.savefig(fname_, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # python 2.7
    names = sorted(glob("./ShoppingMall/data/*.bmp"))
    d1, d2, channels = misc.imread(names[0]).shape
    d1 = 128
    d2 = 160
    num = len(names)
    X = np.zeros((d1, d2, num))
    for n, i in enumerate(names):
        X[:, :, n] = misc.imresize(rgb2gray(misc.imread(i).astype(np.double)) / 255., (d1, d2))

    X = X.reshape(d1 * d2, num)
    clip = 100
    sz = clip
    A, E = PCA_IALM(X[:, : sz])
    A = A.reshape(d1, d2, sz) * 255.
    E = E.reshape(d1, d2, sz) * 255.
    savemat("./ShoppingMall/IALM_background_subtraction.mat", {"1": A, "2": E})
    make_video('IALM')
