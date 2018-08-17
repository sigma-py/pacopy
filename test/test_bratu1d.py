# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import numpy

import pycont


class Bratu1d(object):
    def __init__(self):
        self.n = 51
        h = 1.0 / (self.n - 1)

        self.H = numpy.full(self.n, h)
        self.H[0] = h / 2
        self.H[-1] = h / 2

        self.A = (
            1.0
            / h ** 2
            * scipy.sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(self.n, self.n))
        )
        return

    def inner(self, a, b):
        return numpy.dot(a, self.H * b)

    def f(self, u, lmbda):
        out = self.A.dot(u) + lmbda * numpy.exp(u)
        out[0] = u[0]
        out[-1] = u[-1]
        return out

    def df_dlmbda(self, u, lmbda):
        out = numpy.exp(u)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    def jacobian_solver(self, u, lmbda, rhs):
        M = self.A.copy()
        d = M.diagonal()
        d += lmbda * numpy.exp(u)
        M.setdiag(d)
        # Dirichlet conditions
        assert numpy.all(M.offsets == [-1, 0, 1])
        M.data[0][self.n - 2] = 0.0
        M.data[1][0] = 1.0
        M.data[1][self.n - 1] = 1.0
        M.data[2][1] = 0.0
        return scipy.sparse.linalg.spsolve(M.tocsr(), rhs)


def test_pycont():
    problem = Bratu1d()
    u0 = numpy.zeros(problem.n)
    lmbda0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("λ")
    plt.ylabel("‖u‖_∞")
    lmbda_list = []
    values_list = []
    line1, = ax.plot(lmbda_list, values_list, "-", color="#1f77f4")

    def callback(k, lmbda, sol):
        lmbda_list.append(lmbda)
        line1.set_xdata(lmbda_list)
        values_list.append(numpy.max(numpy.abs(sol)))
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 6.0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        return

    # pycont.natural(problem, u0, lmbda0, callback, max_steps=100)
    pycont.euler_newton(problem, u0, lmbda0, callback, max_steps=500)
    return


if __name__ == "__main__":
    test_pycont()
