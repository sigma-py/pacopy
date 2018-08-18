# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import numpy

import meshio
import meshzoo
import meshplex
import pyfvm
from pyfvm.form_language import integrate, n_dot_grad, dS, Boundary

import pycont


# Just quickly get the diffusion matrix
class Poisson(object):
    def apply(self, u):
        return integrate(lambda x: n_dot_grad(u(x)), dS)

    def dirichlet(self, u):
        return [(lambda x: u(x) - 0.0, Boundary())]


def set_dirichlet_rows(matrix, idx):
    for i in idx:
        matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]] = 0.0
    # Set the diagonal and RHS.
    d = matrix.diagonal()
    d[idx] = 1.0
    matrix.setdiag(d)
    return


class Brusselator2d(object):
    def __init__(self):
        points, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 20, 20)
        self.mesh = meshplex.MeshTri(points, cells)
        self.A, _ = pyfvm.discretize_linear(Poisson(), self.mesh)

        self.a = 4.0
        self.d1 = 1.0
        self.d2 = 2.0
        return

    def inner(self, x, y):
        """Inner product in the domain space (functions and such).
        """
        ux, vx = x
        uy, vy = y
        return numpy.dot(ux, self.mesh.control_volumes * uy) + numpy.dot(
            vx, self.mesh.control_volumes * vy
        )

    def inner_r(self, q, r):
        """Inner product in the range space (residuals and such).
        """
        uq, vq = q
        ur, vr = r
        return numpy.dot(uq, ur) + numpy.dot(vq, vr)

    def f(self, x, b):
        u, v = x
        q = self.d1 * self.A.dot(u) - (b + 1) * u + u ** 2 * v + self.a
        r = self.d2 * self.A.dot(v) + b * u - u ** 2 * v
        # boundary conditions
        i = self.mesh.is_boundary_node
        q[i] = u[i] - self.a
        r[i] = v[i] - b / self.a
        return numpy.array([q, r])

    def df_dlmbda(self, x, b):
        u, v = x
        q = -u
        r = +u
        # boundary conditions
        i = self.mesh.is_boundary_node
        q[i] = 0.0
        r[i] = -1.0 / self.a
        return numpy.array([q, r])

    def jacobian_solver(self, x, b, rhs):
        u, v = x

        # Build the 2x2 block matrix
        #
        #   [d1 A - b+1 + 2uv   u**2       ]
        #   [-2uv + b           d2 A - u**2]
        #
        # and solve it explicitly. There are better ways for doing this, e.g., via the
        # Schur complement.
        ib = self.mesh.is_boundary_node

        A11 = self.d1 * self.A
        diag = A11.diagonal()
        diag += -(b + 1) + 2 * u * v
        A11.setdiag(diag)
        set_dirichlet_rows(A11, numpy.where(ib)[0])

        diag = u ** 2
        diag[ib] = 0.0
        A12 = scipy.sparse.diags([diag], [0])

        diag = -2 * u * v + b
        diag[ib] = 0.0
        A21 = scipy.sparse.diags([diag], [0])

        A22 = self.d2 * self.A
        diag = A22.diagonal()
        diag -= u ** 2
        A22.setdiag(diag)
        set_dirichlet_rows(A22, numpy.where(ib)[0])

        J = scipy.sparse.vstack(
            [scipy.sparse.hstack([A11, A12]), scipy.sparse.hstack([A21, A22])]
        )

        rhs = numpy.concatenate(rhs)
        sol = scipy.sparse.linalg.spsolve(J.tocsr(), rhs)

        n = u.shape[0]
        u_sol = sol[:n]
        v_sol = sol[n:]
        return numpy.array([u_sol, v_sol])


def test_brusselator2d():
    problem = Brusselator2d()
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.zeros((2, n))
    b0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("b")
    plt.ylabel("$||u||_\infty$")
    b_list = []
    values_list = []
    line1, = ax.plot(b_list, values_list, "-", color="#1f77f4")

    def callback(k, b, sol):
        b_list.append(b)
        line1.set_xdata(b_list)
        values_list.append(numpy.max(numpy.abs(sol)))
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 200.0)
        ax.set_ylim(0.0, 40.0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Store the solution
        u, v = sol
        meshio.write_points_cells(
            "sol{:03d}.vtk".format(k),
            problem.mesh.node_coords,
            {"triangle": problem.mesh.cells["nodes"]},
            point_data={"u": u, "v": v},
        )
        return

    # pycont.natural(problem, u0, b0, callback, max_steps=100)
    pycont.euler_newton(problem, u0, b0, callback, max_steps=300)
    return


if __name__ == "__main__":
    test_brusselator2d()
