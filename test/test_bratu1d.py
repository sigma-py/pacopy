# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import numpy

import pacopy


class Bratu1d(object):
    def __init__(self):
        self.n = 51
        h = 1.0 / (self.n - 1)

        self.H = numpy.full(self.n, h)
        self.H[0] = h / 2
        self.H[-1] = h / 2

        self.A = (
            scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(self.n, self.n))
            / h ** 2
        )
        return

    def inner(self, a, b):
        return numpy.dot(a, self.H * b)

    def norm2_r(self, a):
        return numpy.dot(a, a)

    def f(self, u, lmbda):
        out = self.A.dot(u) - lmbda * numpy.exp(u)
        out[0] = u[0]
        out[-1] = u[-1]
        return out

    def df_dlmbda(self, u, lmbda):
        out = -numpy.exp(u)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    def jacobian_solver(self, u, lmbda, rhs):
        M = self.A.copy()
        d = M.diagonal()
        d -= lmbda * numpy.exp(u)
        M.setdiag(d)
        # Dirichlet conditions
        assert numpy.all(M.offsets == [-1, 0, 1])
        M.data[0][self.n - 2] = 0.0
        M.data[1][0] = 1.0
        M.data[1][self.n - 1] = 1.0
        M.data[2][1] = 0.0
        return scipy.sparse.linalg.spsolve(M.tocsr(), rhs)


def test_pacopy():
    problem = Bratu1d()
    u0 = numpy.zeros(problem.n)
    lmbda0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("$||u||_2$")
    ax1.grid()

    ax2 = fig.add_subplot(122)
    ax2.grid()

    lmbda_list = []
    values_list = []
    line1, = ax1.plot(lmbda_list, values_list, "-x", color="#1f77f4")

    line2, = ax2.plot([], [], "-", color="#1f77f4")
    line2.set_xdata(numpy.linspace(0.0, 1.0, problem.n))
    line3, = ax2.plot([], [], "-", color="red")
    line3.set_xdata(numpy.linspace(0.0, 1.0, problem.n))

    def callback(k, lmbda, sol, lmbda_pre, u_pre, du_dlmbda):
        lmbda_list.append(lmbda)
        line1.set_xdata(lmbda_list)
        # values_list.append(numpy.max(numpy.abs(sol)))
        values_list.append(numpy.sqrt(problem.inner(sol, sol)))
        line1.set_ydata(values_list)
        ax1.set_xlim(0.0, 4.0)
        ax1.set_ylim(0.0, 6.0)

        ax1.plot([lmbda_pre], [numpy.sqrt(problem.inner(u_pre, u_pre))], ".r")

        line2.set_ydata(sol)
        line3.set_ydata(du_dlmbda)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 6.0)

        fig.canvas.draw()
        fig.canvas.flush_events()

        input("Press Enter to continue...")
        return

    # pacopy.natural(problem, u0, lmbda0, callback, max_steps=100)

    # The condition number of the Jacobian is about 10^4, so we can only expect Newton
    # to converge up to about this factor above machine precision.
    pacopy.euler_newton(
        problem, u0, lmbda0, callback, max_steps=500, newton_tol=1.0e-10
    )
    return


if __name__ == "__main__":
    test_pacopy()
