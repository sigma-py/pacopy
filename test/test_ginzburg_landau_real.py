# -*- coding: utf-8 -*-
#
"""
The same as GinzburgLandau, except that the complex-valued vectors and operators are
split into real and imaginary part such that all computations are done as float.
This complicates many things, but can be useful for debugging. This formulation is quite
close to the original C++ nosh.
"""
import numpy
import scipy.sparse
from scipy.sparse.linalg import spsolve

import pykry
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
        # <m, edge>
        beta = numpy.einsum("...k,...k->...", magnetic_potential, edge)

        zero = numpy.zeros(edge_ce_ratio.shape, dtype=complex)
        return numpy.array(
            [
                [zero, 1j * edge_ce_ratio * numpy.exp(-1j * beta)],
                [-1j * edge_ce_ratio * numpy.exp(1j * beta), zero],
            ]
        )


def split_sparse_matrix(matrix):
    m = matrix.tocoo()
    data = numpy.concatenate([m.data.real, -m.data.imag, m.data.imag, m.data.real])
    row = numpy.concatenate([2 * m.row, 2 * m.row, 2 * m.row + 1, 2 * m.row + 1])
    col = numpy.concatenate([2 * m.col, 2 * m.col + 1, 2 * m.col, 2 * m.col + 1])
    out = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(2 * m.shape[0], 2 * m.shape[1])
    )
    return out


def to_real(z):
    x = numpy.empty(2 * z.shape[0])
    x[0::2] = z.real
    x[1::2] = z.imag
    return x


def to_complex(x):
    assert x.shape[0] % 2 == 0
    z = x[0::2] + 1j * x[1::2]
    return z


def scalar_multiply(alpha, x):
    return to_real(alpha * to_complex(x))


def multiply(a, b):
    out = numpy.empty(a.shape[0])
    out[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
    out[1::2] = a[0::2] * b[1::2] + a[1::2] * b[0::2]
    return out


def abs2(a):
    out = numpy.zeros(a.shape[0])
    out[0::2] = a[0::2] ** 2 + a[1::2] ** 2
    return out


def conjugate(a):
    out = a.copy()
    out[1::2] *= -1
    return out


def square(a):
    out = a.copy()
    out[0::2] = a[0::2] ** 2 - a[1::2] ** 2
    out[1::2] = 2 * a[0::2] ** 2 - a[1::2] ** 2
    return out


class GinzburgLandauReal(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = -1.0
        self.g = 1.0
        return

    def inner(self, x, y):
        return numpy.dot(x, y)

    def norm2_r(self, q):
        return numpy.dot(q, q)

    def f(self, psi, mu):
        keo = split_sparse_matrix(
            pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        )
        cv = to_real(self.mesh.control_volumes)

        V = numpy.zeros(cv.shape[0])
        V[0::2] = self.V
        out = keo * psi + multiply(cv, multiply(psi, V + self.g * abs2(psi)))

        # Algebraically, The inner product of <f(psi), i*psi> is always 0. We project
        # out that component numerically to avoid convergence failure for the Jacobian
        # updates close to a solution. If this is not done, the Krylov method might hang
        # at something like 10^{-7}.
        # i_psi = to_real(1j * to_complex(psi))
        # out -= self.inner(i_psi, out) / self.inner(i_psi, i_psi) * i_psi

        return out

    def df_dlmbda(self, psi, mu):
        keo_prime = split_sparse_matrix(
            pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[EnergyPrime(mu)])
        )
        out = keo_prime * psi
        # same as in f()
        i_psi = to_real(1j * to_complex(psi))
        out -= self.inner(i_psi, out) / self.inner(i_psi, i_psi) * i_psi
        return out

    def jacobian(self, psi, mu):
        keo = split_sparse_matrix(
            pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        )

        def _apply_jacobian(phi):
            cv = to_real(self.mesh.control_volumes)
            y = keo * phi + multiply(cv, alpha * phi + gPsi0Squared * conjugate(phi))
            return y

        alpha = self.V + self.g * 2.0 * abs2(psi)
        gPsi0Squared = self.g * psi ** 2

        num_unknowns = len(self.mesh.node_coords)
        return pykry.LinearOperator(
            (num_unknowns, num_unknowns),
            complex,
            dot=_apply_jacobian,
            dot_adj=_apply_jacobian,
        )

    def jacobian_solver(self, psi, mu, rhs):
        keo = split_sparse_matrix(
            pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        )
        abs_psi2 = numpy.zeros(psi.shape[0])
        abs_psi2[0::2] += psi[0::2] ** 2 + psi[1::2] ** 2
        cv = to_real(self.mesh.control_volumes)

        def jacobian(psi):
            def _apply_jacobian(phi):
                return keo * phi + alpha * phi + beta * phi.conj()

            alpha = multiply(cv, self.V + self.g * 2.0 * abs_psi2)
            beta = multiply(cv, self.g * multiply(psi, psi))

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (2 * num_unknowns, 2 * num_unknowns),
                float,
                dot=_apply_jacobian,
                dot_adj=_apply_jacobian,
            )

        def prec_inv(psi):
            prec_orig = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
            diag = prec_orig.diagonal()
            diag += self.g * 2.0 * (psi[0::2] ** 2 + psi[1::2] ** 2) * cv[0::2]
            prec_orig.setdiag(diag)
            return split_sparse_matrix(prec_orig).tocsr()

        def prec(psi):
            p = prec_inv(psi)

            def _apply(phi):
                out = spsolve(p, phi)
                return out

            num_unknowns = len(self.mesh.node_coords)
            return pykry.LinearOperator(
                (2 * num_unknowns, 2 * num_unknowns), float, dot=_apply, dot_adj=_apply
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
        return out.xk


def test_self_adjointness():
    problem = GinzburgLandauReal()
    n = problem.mesh.control_volumes.shape[0]
    psi = numpy.random.rand(2 * n)
    jac = problem.jacobian(psi, 0.1)

    for _ in range(1000):
        u = numpy.random.rand(2 * n)
        v = numpy.random.rand(2 * n)
        a0 = problem.inner(u, jac * v)
        a1 = problem.inner(jac * u, v)
        assert abs(a0 - a1) < 1.0e-12
    return


def test_ginzburg_landau(n=20):
    a = 10.0
    points, cells = meshzoo.rectangle(-a / 2, a / 2, -a / 2, a / 2, n, n)
    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandauReal(mesh)
    n = 2 * problem.mesh.control_volumes.shape[0]
    u0 = numpy.ones(n)
    u0[1::2] = 0.0
    mu0 = 0.0

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.axis("square")
    # plt.xlabel("$\\mu$")
    # plt.ylabel("$||\\psi||_2$")
    # plt.grid()
    # b_list = []
    # values_list = []
    # line1, = ax.plot(b_list, values_list, "-", color="#1f77f4")

    # def callback(k, b, sol):
    def callback(k, b, sol):
        # print(problem.inner(sol, sol))
        # b_list.append(b)
        # line1.set_xdata(b_list)
        # values_list.append(numpy.sqrt(problem.inner(sol, sol)))
        # line1.set_ydata(values_list)
        # ax.set_xlim(0.0, 1.0)
        # ax.set_ylim(0.0, 1.0)
        # ax.invert_yaxis()
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # # Store the solution
        # meshio.write_points_cells(
        #     "sol{:03d}.vtk".format(k),
        #     problem.mesh.node_coords,
        #     {"triangle": problem.mesh.cells["nodes"]},
        #     point_data={"psi": numpy.array([numpy.real(sol), numpy.imag(sol)]).T},
        # )
        return

    # pacopy.natural(
    #     problem,
    #     u0,
    #     b0,
    #     callback,
    #     max_steps=1,
    #     lambda_stepsize0=1.0e-2,
    #     newton_max_steps=5,
    #     newton_tol=1.0e-10,
    # )
    pacopy.euler_newton(
        problem,
        u0,
        mu0,
        callback,
        max_steps=2,
        stepsize0=1.0e-2,
        stepsize_max=1.0,
        newton_tol=1.0e-10,
    )
    return


def test_f():
    from test_ginzburg_landau import GinzburgLandau

    a = 10.0
    n = 10
    points, cells = meshzoo.rectangle(-a / 2, a / 2, -a / 2, a / 2, n, n)
    mesh = meshplex.MeshTri(points, cells)

    gl = GinzburgLandau(mesh)
    glr = GinzburgLandauReal(mesh)

    n = points.shape[0]

    numpy.random.seed(123)

    for _ in range(100):
        psi = numpy.random.rand(n) + 1j * numpy.random.rand(n)
        mu = numpy.random.rand(1)[0]
        out = gl.f(psi, mu) * mesh.control_volumes
        out2 = glr.f(to_real(psi), mu)
        assert numpy.all(numpy.abs(out - to_complex(out2)) < 1.0e-12)
    return


if __name__ == "__main__":
    # test_self_adjointness()
    # test_ginzburg_landau(n=20)
    test_f()
