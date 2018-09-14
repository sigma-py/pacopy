# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy
from scipy.sparse.linalg import spsolve

import pykry
import meshio
import meshzoo
import meshplex
import pyfvm
from pyfvm.form_language import dS, n_dot_grad, integrate

import pacopy


# Just quickly get the positive-semidefinite diffusion matrix
class Poisson(object):
    def apply(self, u):
        return integrate(lambda x: -n_dot_grad(u(x)), dS)


class GrossPitaevskii(object):
    """Describes, for example, Bose-Einstein condensates in a trap potential at very low
    temperatures.

    All parameters are chosen as in

       Computing stationary solutions of the two-dimensional Gross–Pitaevskii
       equation with deflated continuation,
       E.G. Charalampidis, P.G.Kevrekidis, P.E.Farrellb,
       <https://doi.org/10.1016/j.cnsns.2017.05.024>,

    except we don't use Dirichlet boundary conditions. Those cause problems with GMRES.
    TODO Find out why.
    """

    def __init__(self):
        a = 24.0
        points, cells = meshzoo.rectangle(-a / 2, a / 2, -a / 2, a / 2, 50, 50)
        self.mesh = meshplex.MeshTri(points, cells)

        x, y, z = self.mesh.node_coords.T
        assert numpy.all(numpy.abs(z) < 1.0e-15)

        self.omega = 0.2
        self.V = 0.5 * self.omega ** 2 * (x ** 2 + y ** 2)

        # For the preconditioner
        assert numpy.all(self.V >= 0)

        self.A, _ = pyfvm.discretize_linear(Poisson(), self.mesh)
        return

    def inner(self, x, y):
        return numpy.real(numpy.dot(x.conj(), self.mesh.control_volumes * y))

    def norm2_r(self, q):
        return numpy.real(numpy.dot(q.conj(), q))

    def f(self, psi, mu):
        out = (
            (0.5 * self.A * psi) / self.mesh.control_volumes
            + (self.V - mu) * psi
            + numpy.abs(psi) ** 2 * psi
        )
        # Dirichlet conditions on the boundary
        # out[self.mesh.is_boundary_node] = psi[self.mesh.is_boundary_node]
        return out

    def df_dlmbda(self, psi, mu):
        out = -psi
        out[self.mesh.is_boundary_node] = 0.0
        return out

    def jacobian_solver(self, psi, mu, rhs):
        def _apply_jacobian(phi):
            out = (
                (0.5 * self.A * phi) / self.mesh.control_volumes
                + (self.V - mu + 2.0 * (psi.real ** 2 + psi.imag ** 2)) * phi
                + psi ** 2 * phi.conj()
            )
            # out[self.mesh.is_boundary_node] = 1.0
            return out

        n = len(self.mesh.node_coords)
        jac = pykry.LinearOperator(
            (n, n), complex, dot=_apply_jacobian, dot_adj=_apply_jacobian
        )

        def prec(psi):
            def _apply(phi):
                prec = 0.5 * self.A
                diag = prec.diagonal()
                cv = self.mesh.control_volumes
                diag += (self.V + 2.0 * (psi.real ** 2 + psi.imag ** 2)) * cv
                prec.setdiag(diag)
                # TODO pyamg solve
                out = spsolve(prec, phi)
                return out

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (num_unknowns, num_unknowns), complex, dot=_apply, dot_adj=_apply
            )

        out = pykry.gmres(
            A=jac,
            b=rhs,
            # M=prec(psi),
            inner_product=self.inner,
            maxiter=100,
            tol=1.0e-12,
            # Minv=prec_inv(psi),
            # U=1j * psi,
        )
        return out.xk


def test_gross_pitaevskii():
    problem = GrossPitaevskii()
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.ones(n, dtype=complex)
    mu0 = 2 * problem.omega

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("$\\mu$")
    plt.ylabel("$||\\psi||_2$ / |\Omega|")
    plt.grid()
    b_list = []
    values_list = []
    line1, = ax.plot(b_list, values_list, "-", color="#1f77f4")

    area = numpy.sum(problem.mesh.control_volumes)

    def callback(k, b, sol):
        b_list.append(b)
        line1.set_xdata(b_list)
        values_list.append(numpy.sqrt(problem.inner(sol, sol)) / area)
        line1.set_ydata(values_list)
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 1.0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Store the solution
        meshio.write_points_cells(
            "sol{:03d}.vtk".format(k),
            problem.mesh.node_coords,
            {"triangle": problem.mesh.cells["nodes"]},
            point_data={"psi": numpy.array([numpy.real(sol), numpy.imag(sol)]).T},
        )
        return

    # pacopy.natural(problem, u0, mu0, callback, max_newton_steps=10)
    pacopy.euler_newton(
        problem, u0, mu0, callback, stepsize0=1.0e-2, max_newton_steps=10
    )
    return


if __name__ == "__main__":
    test_gross_pitaevskii()
