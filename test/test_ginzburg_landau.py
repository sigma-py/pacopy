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

import pacopy


class Energy(object):
    """Specification of the kinetic energy operator.
    """

    def __init__(self, mu):
        super(Energy, self).__init__()
        self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
        self.subdomains = [None]
        return

    def eval(self, mesh, cell_mask):
        nec = mesh.idx_hierarchy[..., cell_mask]
        X = mesh.node_coords[nec]

        edge_midpoint = 0.5 * (X[0] + X[1])
        edge = X[1] - X[0]
        edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = 0.5 * numpy.cross(self.magnetic_field, edge_midpoint)

        # The dot product <magnetic_potential, edge>, executed for many
        # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        beta = numpy.einsum("...k,...k->...", magnetic_potential, edge)

        return numpy.array(
            [
                [edge_ce_ratio, -edge_ce_ratio * numpy.exp(-1j * beta)],
                [-edge_ce_ratio * numpy.exp(1j * beta), edge_ce_ratio],
            ]
        )


class EnergyPrime(object):
    """Derivative by mu.
    """

    def __init__(self, mu):
        super(EnergyPrime, self).__init__()
        self.magnetic_field = mu * numpy.array([0.0, 0.0, 1.0])
        self.subdomains = [None]
        return

    def eval(self, mesh, cell_mask):
        nec = mesh.idx_hierarchy[..., cell_mask]
        X = mesh.node_coords[nec]

        edge_midpoint = 0.5 * (X[0] + X[1])
        edge = X[1] - X[0]
        edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = 0.5 * numpy.cross(self.magnetic_field, edge_midpoint)

        # The dot product <magnetic_potential, edge>, executed for many
        # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        beta = numpy.einsum("...k,...k->...", magnetic_potential, edge)

        return numpy.array(
            [
                [edge_ce_ratio, 1j * edge_ce_ratio * numpy.exp(-1j * beta)],
                [-1j * edge_ce_ratio * numpy.exp(1j * beta), edge_ce_ratio],
            ]
        )


class GinzburgLandau(object):
    def __init__(self):
        points, cells = meshzoo.rectangle(-5.0, 5.0, -5.0, 5.0, 40, 40)
        self.mesh = meshplex.MeshTri(points, cells)

        self.V = -1.0
        self.g = 1.0
        return

    def inner(self, x, y):
        """This is the special Ginzburg-Landau inner product. *bling bling*
        """
        return numpy.real(numpy.dot(x.conj(), self.mesh.control_volumes * y))

    def norm2_r(self, q):
        return numpy.real(numpy.dot(q.conj(), q))

    def f(self, psi, mu):
        keo = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        cv = self.mesh.control_volumes
        return keo * psi + cv * psi * (self.V + self.g * numpy.abs(psi) ** 2)

    def df_dlmbda(self, psi, mu):
        keo_prime = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[EnergyPrime(mu)])
        return keo_prime * psi

    def jacobian_solver(self, psi, mu, rhs):
        keo = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])

        def jacobian(psi):
            def _apply_jacobian(phi):
                cv = self.mesh.control_volumes
                y = keo * phi + cv * alpha * phi + cv * gPsi0Squared * phi.conj()
                return y

            alpha = self.V + self.g * 2.0 * (psi.real ** 2 + psi.imag ** 2)
            gPsi0Squared = self.g * psi ** 2

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (num_unknowns, num_unknowns),
                complex,
                dot=_apply_jacobian,
                dot_adj=_apply_jacobian,
            )

        def prec(psi):
            def _apply(phi):
                prec = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
                # Add diagonal to avoid singularity for mu = 0.
                diag = prec.diagonal()
                cv = self.mesh.control_volumes
                diag += cv * self.g * 2.0 * (psi.real ** 2 + psi.imag ** 2)
                prec.setdiag(diag)
                # TODO pyamg solve
                out = spsolve(prec, phi)
                return out.reshape(phi.shape)

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (num_unknowns, num_unknowns), complex, dot=_apply, dot_adj=_apply
            )

        jac = jacobian(psi)
        prec = prec(psi)
        out = pykry.gmres(
            A=jac, b=rhs, M=prec, inner_product=self.inner, maxiter=1000, tol=1.0e-8
        )
        return out.xk


def test_ginzburg_landau():
    problem = GinzburgLandau()
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.ones(n, dtype=complex)
    b0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("$\\mu$")
    plt.ylabel("$||\\psi||_2$")
    plt.grid()
    b_list = []
    values_list = []
    line1, = ax.plot(b_list, values_list, "-", color="#1f77f4")

    def callback(k, b, sol, a_, b_, c_):
        print(problem.inner(sol, sol))
        b_list.append(b)
        line1.set_xdata(b_list)
        values_list.append(numpy.sqrt(problem.inner(sol, sol)))
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
            point_data={"psi": numpy.array([numpy.real(sol), numpy.imag(sol)]).T},
        )
        return

    pacopy.natural(
        problem,
        u0,
        b0,
        callback,
        max_steps=100,
        lambda_stepsize0=1.0e-2,
        newton_max_steps=5,
        newton_tol=1.0e-10,
    )
    # pacopy.euler_newton(problem, u0, b0, callback, max_steps=100, stepsize0=1.0e-2)
    return


if __name__ == "__main__":
    test_ginzburg_landau()
