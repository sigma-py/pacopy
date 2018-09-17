# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy
from scipy.sparse.linalg import spsolve

import meshio
import meshzoo
import meshplex
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS

import pacopy


# Just quickly get the diffusion matrix
class Poisson(object):
    def apply(self, u):
        return integrate(lambda x: n_dot_grad(u(x)), dS)


def set_dirichlet_rows(matrix, idx):
    for i in idx:
        matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
    d = matrix.diagonal()
    d[idx] = 1.0
    matrix.setdiag(d)
    return


class AllenCahn(object):
    def __init__(self):
        points, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 30, 30)
        self.mesh = meshplex.MeshTri(points, cells)
        # This matrix self.A is negative semidefinite
        self.A, _ = pyfvm.discretize_linear(Poisson(), self.mesh)
        tol = 1.0e-12
        self.idx_left = numpy.where(self.mesh.node_coords[:, 0] < tol)[0]
        self.idx_right = numpy.where(self.mesh.node_coords[:, 0] > 1.0 - tol)[0]
        self.idx_bottom = numpy.where(self.mesh.node_coords[:, 1] < tol)[0]
        self.idx_top = numpy.where(self.mesh.node_coords[:, 1] > 1.0 - tol)[0]
        return

    def inner(self, x, y):
        return numpy.dot(x, y)

    def norm2_r(self, q):
        return numpy.dot(q, q)

    def f(self, u, delta):
        cv = self.mesh.control_volumes
        out = -delta * self.A * u - cv / delta * (1 - u ** 2) * u
        # Dirichlet boundary conditions
        out[self.idx_left] = u[self.idx_left] - 1.0
        out[self.idx_right] = u[self.idx_right] - 1.0
        out[self.idx_bottom] = u[self.idx_bottom] + 1.0
        out[self.idx_top] = u[self.idx_top] + 1.0
        return out

    def df_dlmbda(self, u, delta):
        cv = self.mesh.control_volumes
        out = -self.A * u + cv / delta ** 2 * (1 - u ** 2) * u
        out[self.idx_left] = 0.0
        out[self.idx_right] = 0.0
        out[self.idx_bottom] = 0.0
        out[self.idx_top] = 0.0
        return out

    def jacobian(self, u, delta):
        cv = self.mesh.control_volumes
        jac = -delta * self.A
        diag = jac.diagonal()
        diag -= cv / delta * (1 - 3 * u ** 2)
        jac.setdiag(diag)
        set_dirichlet_rows(jac, self.idx_left)
        set_dirichlet_rows(jac, self.idx_right)
        set_dirichlet_rows(jac, self.idx_top)
        set_dirichlet_rows(jac, self.idx_bottom)
        return jac

    def jacobian_solver(self, u, delta, rhs):
        jac = self.jacobian(u, delta)
        out = spsolve(jac, rhs)
        return out


def test_jacobian():
    # assert that the jacobian is correctly defined
    problem = AllenCahn()
    n = problem.mesh.control_volumes.shape[0]

    delta = 0.04
    eps = 1.0e-5
    for _ in range(100):
        u = numpy.random.rand(n)
        v = numpy.random.rand(n)
        out0 = (problem.f(u + eps * v, delta) - problem.f(u - eps * v, delta)) / (
            2 * eps
        )
        out1 = problem.jacobian(u, delta) * v
        assert numpy.all(numpy.abs(out0 - out1) < 1.0e-10)
    return


def test_allen_cahn():
    problem = AllenCahn()
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.zeros(n, dtype=float)
    # delta0 = 0.04
    delta0 = 0.2

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("$\\mu$")
    plt.ylabel("$||u||_2^2 / |\Omega|$")
    plt.grid()
    b_list = []
    values_list = []
    line1, = ax.plot(b_list, values_list, "-", color="#1f77f4")

    area = numpy.sum(problem.mesh.control_volumes)

    def callback(k, b, sol):
        b_list.append(b)
        line1.set_xdata(b_list)
        values_list.append(problem.inner(sol, sol) / area)
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.invert_yaxis()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Store the solution
        meshio.write_points_cells(
            "sol{:03d}.vtk".format(k),
            problem.mesh.node_coords,
            {"triangle": problem.mesh.cells["nodes"]},
            point_data={"u": sol},
        )
        # input("Press")
        return

    pacopy.natural(
        problem,
        u0,
        delta0,
        callback,
        max_steps=6,
        lambda_stepsize0=-1.0e-1,
        max_newton_steps=16,
        newton_tol=1.0e-10,
    )
    # pacopy.euler_newton(
    #     problem,
    #     u0,
    #     delta0,
    #     callback,
    #     max_steps=10,
    #     stepsize0=-1.0e-1,
    #     stepsize_max=1.0,
    #     newton_tol=1.0e-10,
    # )
    return


if __name__ == "__main__":
    test_allen_cahn()
    # test_jacobian()
