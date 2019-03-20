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

    def jacobian(self, u, lmbda):
        M = self.A.copy()
        d = M.diagonal().copy()
        d -= lmbda * numpy.exp(u)
        M.setdiag(d)
        # Dirichlet conditions
        assert numpy.all(M.offsets == [-1, 0, 1])
        M.data[0][self.n - 2] = 0.0
        M.data[1][0] = 1.0
        M.data[1][self.n - 1] = 1.0
        M.data[2][1] = 0.0
        return M.tocsr()

    def jacobian_solver(self, u, lmbda, rhs):
        return scipy.sparse.linalg.spsolve(self.jacobian(u, lmbda), rhs)


class RangeException(Exception):
    pass


def test_bratu(max_steps=10, update_plot=False):
    problem = Bratu1d()
    u0 = numpy.zeros(problem.n)
    lmbda0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("$||u||_2$")
    ax1.grid()

    # ax2 = fig.add_subplot(122)
    # ax2.grid()

    line1, = ax1.plot([], [], "-x", color="#1f77f4")

    # line2, = ax2.plot([], [], "-", color="#1f77f4")
    # line2.set_xdata(numpy.linspace(0.0, 1.0, problem.n))
    # line3, = ax2.plot([], [], "-", color="red")
    # line3.set_xdata(numpy.linspace(0.0, 1.0, problem.n))

    def callback(k, lmbda, sol):
        if update_plot:
            line1.set_xdata(numpy.append(line1.get_xdata(), lmbda))
            # val = numpy.max(numpy.abs(sol))
            val = numpy.sqrt(problem.inner(sol, sol))
            line1.set_ydata(numpy.append(line1.get_ydata(), val))

            ax1.set_xlim(0.0, 4.0)
            ax1.set_ylim(0.0, 6.0)

            if ax1.get_ylim()[1] < val:
                raise RangeException

            # ax1.plot([lmbda_pre], [numpy.sqrt(problem.inner(u_pre, u_pre))], ".r")

            # line2.set_ydata(sol)
            # line3.set_ydata(du_dlmbda)
            # ax2.set_xlim(0.0, 1.0)
            # ax2.set_ylim(0.0, 6.0)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.savefig('bratu1d.png'.format(k), transparent=True, bbox_inches="tight")
        return

    pacopy.natural(problem, u0, lmbda0, callback, max_steps=max_steps,
                   newton_tol=1e-10, milestones=numpy.arange(.5, 3.2, .5))

    # The condition number of the Jacobian is about 10^4, so we can only expect Newton
    # to converge up to about this factor above machine precision.
    try:
        pacopy.euler_newton(
            problem, u0, lmbda0, callback, max_steps=max_steps, newton_tol=1.0e-10
        )
    except RangeException:
        if update_plot:
            plt.pause(5)
    return


if __name__ == "__main__":
    test_bratu(500, update_plot=True)
    # test_self_adjointness()
