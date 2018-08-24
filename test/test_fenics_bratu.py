# -*- coding: utf-8 -*-
#
import math

import matplotlib.pyplot as plt
from dolfin import (
    UnitSquareMesh,
    FunctionSpace,
    dx,
    assemble,
    dot,
    grad,
    DirichletBC,
    TrialFunction,
    TestFunction,
    exp,
    Function,
    solve,
    XDMFFile,
)

import pacopy


class Bratu(object):
    def __init__(self):
        self.mesh = UnitSquareMesh(40, 40, "left/right")

        self.V = FunctionSpace(self.mesh, "Lagrange", 1)

        self.bc = DirichletBC(self.V, 0.0, "on_boundary")

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
        out = self.a * u - lmbda * assemble(exp(ufun) * v * dx)
        DirichletBC(self.V, ufun, "on_boundary").apply(out)
        return out

    def df_dlmbda(self, u, lmbda):
        v = TestFunction(self.V)
        ufun = Function(self.V)
        ufun.vector()[:] = u
        out = -assemble(exp(ufun) * v * dx)
        self.bc.apply(out)
        return out

    def jacobian_solver(self, u, lmbda, rhs):
        t = TrialFunction(self.V)
        v = TestFunction(self.V)
        # from dolfin import Constant
        # a = assemble(
        #     dot(grad(t), grad(v)) * dx - Constant(lmbda) * exp(u) * t * v * dx
        # )
        ufun = Function(self.V)
        ufun.vector()[:] = u
        a = self.a - lmbda * assemble(exp(ufun) * t * v * dx)
        self.bc.apply(a)
        x = Function(self.V)
        solve(a, x.vector(), rhs, "gmres", "ilu")
        return x.vector()


def test_bratu_fenics():
    problem = Bratu()
    u0 = Function(problem.V).vector()
    lmbda0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("$\\lambda$")
    plt.ylabel("$||u||_2$")
    plt.grid()
    lmbda_list = []
    values_list = []
    line1, = ax.plot(lmbda_list, values_list, "-", color="#1f77f4")

    f = XDMFFile("sol.xdmf")
    u = Function(problem.V)

    def callback(k, lmbda, sol, lmbda_pre, u_pre):
        lmbda_list.append(lmbda)
        line1.set_xdata(lmbda_list)
        values_list.append(math.sqrt(problem.inner(sol, sol)))
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 6.0)

        import numpy

        ax.plot([lmbda_pre], [numpy.sqrt(problem.inner(u_pre, u_pre))], ".r")

        fig.canvas.draw()
        fig.canvas.flush_events()

        u.vector()[:] = sol
        f.write(u, k)
        return

    # pacopy.natural(problem, u0, lmbda0, callback, max_steps=100)
    pacopy.euler_newton(
        problem, u0, lmbda0, callback, max_steps=500, newton_tol=1.0e-10
    )
    return


if __name__ == "__main__":
    test_bratu_fenics()
