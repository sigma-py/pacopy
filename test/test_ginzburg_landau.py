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
        self.dmagnetic_field_dmu = numpy.array([0.0, 0.0, 1.0])
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
        # <m, edge>
        beta = numpy.einsum("...k,...k->...", magnetic_potential, edge)

        # project the magnetic potential on the edge at the midpoint
        dmagnetic_potential_dmu = 0.5 * numpy.cross(
            self.dmagnetic_field_dmu, edge_midpoint
        )
        # <m, edge>
        dbeta_dmu = numpy.einsum("...k,...k->...", dmagnetic_potential_dmu, edge)

        zero = numpy.zeros(edge_ce_ratio.shape, dtype=complex)
        return numpy.array(
            [
                [zero, 1j * dbeta_dmu * edge_ce_ratio * numpy.exp(-1j * beta)],
                [-1j * dbeta_dmu * edge_ce_ratio * numpy.exp(1j * beta), zero],
            ]
        )


class GinzburgLandau(object):
    def __init__(self, mesh):
        self.mesh = mesh

        self.V = -1.0
        self.g = 1.0

        # import matplotlib.pyplot as plt
        # self.fig1, self.ax1 = plt.subplots()
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
        out = (keo * psi) / cv + psi * (self.V + self.g * numpy.abs(psi) ** 2)

        # Algebraically, The inner product of <f(psi), i*psi> is always 0. We project
        # out that component numerically to avoid convergence failure for the Jacobian
        # updates close to a solution. If this is not done, the Krylov method might hang
        # at something like 10^{-7}.
        i_psi = 1j * psi
        out -= self.inner(i_psi, out) / self.inner(i_psi, i_psi) * i_psi

        return out

    def df_dlmbda(self, psi, mu):
        keo_prime = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[EnergyPrime(mu)])
        out = (keo_prime * psi) / self.mesh.control_volumes
        # same as in f()
        i_psi = 1j * psi
        out -= self.inner(i_psi, out) / self.inner(i_psi, i_psi) * i_psi
        return out

    def jacobian_solver(self, psi, mu, rhs):
        keo = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        cv = self.mesh.control_volumes

        def jacobian(psi):
            def _apply_jacobian(phi):
                return (keo * phi) / cv + alpha * phi + beta * phi.conj()

            alpha = self.V + self.g * 2.0 * (psi.real ** 2 + psi.imag ** 2)
            beta = self.g * psi ** 2

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (num_unknowns, num_unknowns),
                complex,
                dot=_apply_jacobian,
                dot_adj=_apply_jacobian,
            )

        def prec_inv(psi):
            prec = keo.copy()
            # Add diagonal to avoid singularity for mu = 0. Also, this is a better
            # preconditioner.
            diag = prec.diagonal()
            diag += self.g * 2.0 * (psi.real ** 2 + psi.imag ** 2) * cv
            prec.setdiag(diag)
            return prec

        def prec(psi):
            p = prec_inv(psi)

            def _apply(phi):
                # ml = pyamg.smoothed_aggregation_solver(p, phi)
                # out = ml.solve(b=phi, tol=1e-12)
                out = spsolve(p, phi)
                return out

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (num_unknowns, num_unknowns), complex, dot=_apply, dot_adj=_apply
            )

        jac = jacobian(psi)
        out = pykry.gmres(
            A=jac,
            b=rhs,
            M=prec(psi),
            inner_product=self.inner,
            maxiter=100,
            tol=1.0e-12,
            # Minv=prec_inv(psi),
            # U=1j * psi,
        )
        # print("Krylov iterations:", out.iter)
        # print("Krylov residual:", out.resnorms[-1])
        # res = jac * out.xk - rhs
        # print("Krylov residual (explicit):", numpy.sqrt(self.norm2_r(res)))

        # self.ax1.semilogy(out.resnorms)
        # self.ax1.grid()
        # plt.show()

        # Since
        #
        #     (J_psi) psi = K psi + (-1 +2|psi|^2) psi - psi^2 conj(psi)
        #                 = K psi - psi + psi |psi|^2
        #                 = f(psi),
        #
        # we have (J_psi)(i psi) = i f(psi). The RHS is typically f(psi) or
        # df/dlmbda(psi), but obviously
        #
        #     <f(psi), (J_psi)(i psi)> = <f(psi), i f(psi)> = 0,
        #
        # so the i*psi-component in the solution plays no role if the rhs is f(psi).
        # Using 0 as a starting guess for Krylov, the solution will have no component in
        # the i*psi-direction. This means that the Newton updates won't jump around the
        # solution manifold. It wouldn't matter if they did, though.
        # TODO show this for df/dlmbda as well
        # i_psi = 1j * psi
        # out.xk -= self.inner(i_psi, out.xk) / self.inner(i_psi, i_psi) * i_psi
        # print("solution component i*psi", self.inner(i_psi, out.xk) / numpy.sqrt(self.inner(i_psi, i_psi)))
        return out.xk


def test_self_adjointness():
    problem = GinzburgLandau()
    n = problem.mesh.control_volumes.shape[0]
    numpy.random.seed(0)

    for _ in range(100):
        mu = numpy.random.rand(1)
        psi = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        jac = problem.jacobian(psi, mu)
        u = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        v = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        a0 = problem.inner(u, jac * v)
        a1 = problem.inner(jac * u, v)
        assert abs(a0 - a1) < 1.0e-12

    return


def test_f_i_psi():
    """Assert that <f(psi), i psi> == 0.
    """
    problem = GinzburgLandau()
    n = problem.mesh.control_volumes.shape[0]

    numpy.random.seed(0)

    for _ in range(100):
        mu = numpy.random.rand(1)
        psi = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        f = problem.f(psi, mu)
        assert abs(problem.inner(1j * psi, f)) < 1.0e-13

    return


def test_df_dlmbda():
    points, cells = meshzoo.rectangle(-5.0, 5.0, -5.0, 5.0, 30, 30)
    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandau(mesh)
    n = problem.mesh.control_volumes.shape[0]
    numpy.random.seed(0)

    for _ in range(100):
        mu = numpy.random.rand(1)
        psi = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        out = problem.df_dlmbda(psi, mu)

        # finite difference
        eps = 1.0e-5
        diff = (problem.f(psi, mu + eps) - problem.f(psi, mu - eps)) / (2 * eps)
        nrm = numpy.dot((out - diff).conj(), out - diff).real
        assert nrm < 1.0e-12

    return


def test_ginzburg_landau():
    a = 10.0
    points, cells = meshzoo.rectangle(-a / 2, a / 2, -a / 2, a / 2, 50, 50)
    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandau(mesh)
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.ones(n, dtype=complex)
    b0 = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("square")
    plt.xlabel("$\\mu$")
    plt.ylabel("$||\\psi||_2^2 / |\Omega|$")
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
            point_data={"psi": numpy.array([numpy.real(sol), numpy.imag(sol)]).T},
        )
        # input("Press")
        return

    # pacopy.natural(
    #     problem,
    #     u0,
    #     b0,
    #     callback,
    #     max_steps=1000,
    #     lambda_stepsize0=1.0e-2,
    #     newton_max_steps=5,
    #     newton_tol=1.0e-10,
    # )
    pacopy.euler_newton(
        problem,
        u0,
        b0,
        callback,
        max_steps=10,
        stepsize0=1.0e-2,
        stepsize_max=1.0,
        newton_tol=1.0e-10,
    )
    return


if __name__ == "__main__":
    # test_self_adjointness()
    # test_f_i_psi()
    test_ginzburg_landau()
    # test_df_dlmbda()
