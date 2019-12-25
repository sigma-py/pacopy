import math

import matplotlib.pyplot as plt
import pytest

import pacopy

dolfin = pytest.importorskip("dolfin")


def test_mittelmann_fenics():
    from dolfin import (
        RectangleMesh,
        FunctionSpace,
        dx,
        assemble,
        dot,
        grad,
        TrialFunction,
        TestFunction,
        exp,
        Function,
        solve,
        Point,
    )

    class Mittelmann(object):
        def __init__(self):
            mesh = RectangleMesh(Point(-0.5, -0.5), Point(+0.5, +0.5), 20, 20)

            self.V = FunctionSpace(mesh, "Lagrange", 1)

            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            self.a = assemble(dot(grad(u), grad(v)) * dx)
            self.m = assemble(u * v * dx)
            return

        def inner(self, a, b):
            return a.inner(self.m * b)

        def norm2_r(self, a):
            return a.inner(a)

        def f(self, u, lmbda):
            v = TestFunction(self.V)
            ufun = Function(self.V)
            ufun.vector()[:] = u
            out = (
                self.a * u
                - 10 * assemble(ufun * v * dx)
                + 10 * lmbda * assemble(exp(ufun) * v * dx)
            )
            return out

        def df_dlmbda(self, u, lmbda):
            v = TestFunction(self.V)
            ufun = Function(self.V)
            ufun.vector()[:] = u
            out = 10 * assemble(exp(ufun) * v * dx)
            return out

        def jacobian_solver(self, u, lmbda, rhs):
            t = TrialFunction(self.V)
            v = TestFunction(self.V)
            ufun = Function(self.V)
            ufun.vector()[:] = u
            a = (
                self.a
                - 10 * assemble(t * v * dx)
                + 10 * lmbda * assemble(exp(ufun) * t * v * dx)
            )
            x = Function(self.V)
            solve(a, x.vector(), rhs)
            return x.vector()

    problem = Mittelmann()
    u0 = Function(problem.V).vector()
    lmbda0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.axis("square")
    plt.xlabel("$\\lambda$")
    plt.ylabel("$||u||_2$")
    plt.grid()
    lmbda_list = []
    values_list = []
    (line1,) = ax.plot(lmbda_list, values_list, "-", color="#1f77f4")

    def callback(k, lmbda, sol):
        lmbda_list.append(lmbda)
        line1.set_xdata(lmbda_list)
        values_list.append(math.sqrt(problem.inner(sol, sol)))
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 5.0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        return

    # pacopy.natural(problem, u0, lmbda0, callback, max_steps=100)
    pacopy.euler_newton(problem, u0, lmbda0, callback, max_steps=500)
    return


if __name__ == "__main__":
    test_mittelmann_fenics()
