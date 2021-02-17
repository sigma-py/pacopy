"""
The same as GinzburgLandau, except that the complex-valued vectors and operators are
split into real and imaginary part such that all computations are done as float.
This complicates many things, but can be useful for debugging. This formulation is quite
close to the original C++ nosh.
"""
import meshio
import meshplex
import meshzoo
import numpy as np
import pyfvm
import pykry
import scipy.sparse
import yaml
from scipy.sparse.linalg import spsolve

import pacopy


class Energy:
    """Specification of the kinetic energy operator."""

    def __init__(self, mu):
        self.magnetic_field = mu * np.array([0.0, 0.0, 1.0])
        self.subdomains = [None]

    def eval(self, mesh, cell_mask):
        nec = mesh.idx_hierarchy[..., cell_mask]
        X = mesh.points[nec]

        edge_midpoint = 0.5 * (X[0] + X[1])
        edge = X[1] - X[0]
        edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = 0.5 * np.cross(self.magnetic_field, edge_midpoint)

        # The dot product <magnetic_potential, edge>, executed for many
        # points at once; cf. <http://stackoverflow.com/a/26168677/353337>.
        beta = np.einsum("...k,...k->...", magnetic_potential, edge)

        return np.array(
            [
                [edge_ce_ratio, -edge_ce_ratio * np.exp(-1j * beta)],
                [-edge_ce_ratio * np.exp(1j * beta), edge_ce_ratio],
            ]
        )


class EnergyPrime:
    """Derivative by mu."""

    def __init__(self, mu):
        self.magnetic_field = mu * np.array([0.0, 0.0, 1.0])
        self.dmagnetic_field_dmu = np.array([0.0, 0.0, 1.0])
        self.subdomains = [None]

    def eval(self, mesh, cell_mask):
        nec = mesh.idx_hierarchy[..., cell_mask]
        X = mesh.points[nec]

        edge_midpoint = 0.5 * (X[0] + X[1])
        edge = X[1] - X[0]
        edge_ce_ratio = mesh.ce_ratios[..., cell_mask]

        # project the magnetic potential on the edge at the midpoint
        magnetic_potential = 0.5 * np.cross(self.magnetic_field, edge_midpoint)
        # <m, edge>
        beta = np.einsum("...k,...k->...", magnetic_potential, edge)

        # project the magnetic potential on the edge at the midpoint
        dmagnetic_potential_dmu = 0.5 * np.cross(
            self.dmagnetic_field_dmu, edge_midpoint
        )
        # <m, edge>
        dbeta_dmu = np.einsum("...k,...k->...", dmagnetic_potential_dmu, edge)

        zero = np.zeros(edge_ce_ratio.shape, dtype=complex)
        return np.array(
            [
                [zero, 1j * dbeta_dmu * edge_ce_ratio * np.exp(-1j * beta)],
                [-1j * dbeta_dmu * edge_ce_ratio * np.exp(1j * beta), zero],
            ]
        )


def split_sparse_matrix(matrix):
    m = matrix.tocoo()
    data = np.concatenate([m.data.real, -m.data.imag, m.data.imag, m.data.real])
    row = np.concatenate([2 * m.row, 2 * m.row, 2 * m.row + 1, 2 * m.row + 1])
    col = np.concatenate([2 * m.col, 2 * m.col + 1, 2 * m.col, 2 * m.col + 1])
    out = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(2 * m.shape[0], 2 * m.shape[1])
    )
    return out


def to_real(z):
    x = np.empty(2 * z.shape[0])
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
    out = np.empty(a.shape[0])
    out[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
    out[1::2] = a[0::2] * b[1::2] + a[1::2] * b[0::2]
    return out


def abs2(a):
    out = np.zeros(a.shape[0])
    out[0::2] = a[0::2] ** 2 + a[1::2] ** 2
    return out


def conjugate(a):
    out = a.copy()
    out[1::2] *= -1
    return out


def square(a):
    out = a.copy()
    out[0::2] = a[0::2] ** 2 - a[1::2] ** 2
    out[1::2] = 2 * a[0::2] * a[1::2]
    return out


class GinzburgLandauReal:
    def __init__(self, mesh):
        self.mesh = mesh
        self.V = -1.0
        self.g = 1.0

    def inner(self, x, y):
        return np.dot(x, y)

    def norm2_r(self, q):
        return np.dot(q, q)

    def f(self, psi, mu):
        keo = split_sparse_matrix(
            pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        )
        cv = to_real(self.mesh.control_volumes)

        V = np.zeros(cv.shape[0])
        V[0::2] = self.V
        out = keo * psi + multiply(cv, multiply(psi, V + self.g * abs2(psi)))

        # Algebraically, The inner product of <f(psi), i*psi> is always 0. We project
        # out that component numerically to avoid convergence failure for the Jacobian
        # updates close to a solution. If this is not done, the Krylov method might hang
        # at something like 10^{-7}.
        i_psi = to_real(1j * to_complex(psi))
        out -= self.inner(i_psi, out) / self.inner(i_psi, i_psi) * i_psi

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
        cv = to_real(self.mesh.control_volumes)

        # cv * alpha
        V = np.zeros(cv.shape[0])
        V[0::2] = self.V
        alpha = V + self.g * 2.0 * abs2(psi)
        jac = keo.copy()
        diag0 = jac.diagonal(0)
        b = multiply(cv, alpha)[0::2]
        diag0[0::2] += b
        diag0[1::2] += b
        jac.setdiag(diag0, k=0)

        # cv * beta
        beta = self.g * square(psi)
        diag0 = jac.diagonal(0)
        b = multiply(cv, beta)[0::2]
        diag0[0::2] += b
        diag0[1::2] -= b
        jac.setdiag(diag0, k=0)
        #
        b = multiply(cv, beta)[1::2]
        diag1 = jac.diagonal(1)
        diag1[0::2] += b
        jac.setdiag(diag1, k=1)
        #
        diag2 = jac.diagonal(-1)
        diag2[0::2] += b
        jac.setdiag(diag2, k=-1)

        return jac

    def jacobian_solver(self, psi, mu, rhs):
        abs_psi2 = np.zeros(psi.shape[0])
        abs_psi2[0::2] += psi[0::2] ** 2 + psi[1::2] ** 2
        cv = to_real(self.mesh.control_volumes)

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

            num_unknowns = len(self.mesh.points)
            return pykry.LinearOperator(
                (2 * num_unknowns, 2 * num_unknowns), float, dot=_apply, dot_adj=_apply
            )

        jac = self.jacobian(psi, mu)

        # Cannot use direct solve since jacobian is always singular
        # return spsolve(jac, rhs)

        out = pykry.gmres(
            A=jac,
            b=rhs,
            # TODO enable preconditioner
            # M=prec(psi),
            inner_product=self.inner,
            maxiter=100,
            tol=1.0e-12,
        )
        return out.xk

    def jacobian_eigenvalue(self, psi, mu):
        jac = self.jacobian(psi, mu)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(jac, 2, which="SM")
        # ev = np.sort(np.linalg.eigvalsh(jac.toarray()))
        # One of the eigenvalues is the one corresponding to psi*i. Filter it out.
        ipsi = scalar_multiply(1j, psi) / np.sqrt(self.inner(psi, psi))
        # normalize eigenvectors
        eigvecs[:, 0] /= np.sqrt(self.inner(eigvecs[:, 0], eigvecs[:, 0]))
        eigvecs[:, 1] /= np.sqrt(self.inner(eigvecs[:, 1], eigvecs[:, 1]))
        ev0 = self.inner(eigvecs[:, 0], ipsi)
        ev1 = self.inner(eigvecs[:, 1], ipsi)

        if np.all(np.abs(eigvals) < 1.0e-10):
            # If all eigenvalues are (close to) 0, the eigenvectors are a base of a
            # 2-dimensional subspace. None of the base vectors necessarily i*psi that
            # needs to be filtered out. For this reason, take the one that is closest to
            # i*psi, kick it out, and remove the i*psi part from all other vectors.
            if ev0 > ev1:
                eigvecs[:, 1] -= self.inner(ipsi, eigvecs[:, 1]) * ipsi
                out = eigvals[1], eigvecs[:, 1]
            else:
                eigvecs[:, 0] -= self.inner(ipsi, eigvecs[:, 0]) * ipsi
                out = eigvals[0], eigvecs[:, 0]
        else:
            if abs(1.0 - abs(ev0)) < 1.0e-10:
                out = eigvals[1], eigvecs[:, 1]
            else:
                assert abs(1.0 - abs(ev1)) < 1.0e-10
                out = eigvals[0], eigvecs[:, 0]

        return out


def test_self_adjointness():
    a = 10.0
    n = 10
    points, cells = meshzoo.rectangle_tri((-a / 2, -a / 2), (a / 2, a / 2), n)
    # add column with zeros for magnetic potential
    points = np.column_stack([points, np.zeros(points.shape[0])])

    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandauReal(mesh)
    n = problem.mesh.control_volumes.shape[0]
    psi = np.random.rand(2 * n)
    jac = problem.jacobian(psi, 0.1)

    for _ in range(1000):
        u = np.random.rand(2 * n)
        v = np.random.rand(2 * n)
        a0 = problem.inner(u, jac * v)
        a1 = problem.inner(jac * u, v)
        assert abs(a0 - a1) < 1.0e-12


def test_f():
    from test_ginzburg_landau import GinzburgLandau

    a = 10.0
    n = 10
    points, cells = meshzoo.rectangle_tri((-a / 2, -a / 2), (a / 2, a / 2), n)
    # add column with zeros for magnetic potential
    points = np.column_stack([points, np.zeros(points.shape[0])])

    mesh = meshplex.MeshTri(points, cells)

    gl = GinzburgLandau(mesh)
    glr = GinzburgLandauReal(mesh)

    n = points.shape[0]

    np.random.seed(123)

    for _ in range(10):
        psi = np.random.rand(n) + 1j * np.random.rand(n)
        mu = np.random.rand(1)[0]
        out = gl.f(psi, mu) * mesh.control_volumes
        out2 = glr.f(to_real(psi), mu)
        assert np.all(np.abs(out - to_complex(out2)) < 1.0e-12)


def test_df_dlmbda():
    from test_ginzburg_landau import GinzburgLandau

    a = 10.0
    n = 10
    points, cells = meshzoo.rectangle_tri((-a / 2, -a / 2), (a / 2, a / 2), n)
    # add column with zeros for magnetic potential
    points = np.column_stack([points, np.zeros(points.shape[0])])

    mesh = meshplex.MeshTri(points, cells)

    gl = GinzburgLandau(mesh)
    glr = GinzburgLandauReal(mesh)

    n = points.shape[0]

    np.random.seed(123)

    for _ in range(10):
        psi = np.random.rand(n) + 1j * np.random.rand(n)
        mu = np.random.rand(1)[0]
        out = gl.df_dlmbda(psi, mu) * mesh.control_volumes
        out2 = glr.df_dlmbda(to_real(psi), mu)
        assert np.all(np.abs(out - to_complex(out2)) < 1.0e-12)


def test_jacobian():
    from test_ginzburg_landau import GinzburgLandau

    a = 10.0
    n = 10
    points, cells = meshzoo.rectangle_tri((-a / 2, -a / 2), (a / 2, a / 2), n)
    # add column with zeros for magnetic potential
    points = np.column_stack([points, np.zeros(points.shape[0])])

    mesh = meshplex.MeshTri(points, cells)

    gl = GinzburgLandau(mesh)
    glr = GinzburgLandauReal(mesh)

    n = points.shape[0]

    np.random.seed(123)

    for _ in range(10):
        psi = np.random.rand(n) + 1j * np.random.rand(n)
        mu = np.random.rand(1)[0]
        jac0 = gl.jacobian(psi, mu)
        jac1 = glr.jacobian(to_real(psi), mu)
        for _ in range(10):
            phi = np.random.rand(n) + 1j * np.random.rand(n)
            out0 = (jac0 * phi) * mesh.control_volumes
            out1 = to_complex(jac1 * to_real(phi))
            assert np.all(np.abs(out0 - out1) < 1.0e-12)


def test_continuation(max_steps=5):
    a = 10.0
    n = 20
    points, cells = meshzoo.rectangle_tri((-a / 2, -a / 2), (a / 2, a / 2), n)
    # add column with zeros for magnetic potential
    points = np.column_stack([points, np.zeros(points.shape[0])])

    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandauReal(mesh)
    num_unknowns = 2 * problem.mesh.control_volumes.shape[0]
    u0 = np.ones(num_unknowns)
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

    mu_list = []

    filename = "sol.xdmf"
    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(
            problem.mesh.points, [("triangle", problem.mesh.cells["points"])]
        )

        def callback(k, mu, sol):
            mu_list.append(mu)
            # Store the solution
            psi = np.array([sol[0::2], sol[1::2]]).T
            writer.write_data(k, point_data={"psi": psi})
            with open("data.yml", "w") as fh:
                yaml.dump({"filename": filename, "mu": [float(m) for m in mu_list]}, fh)

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
            max_steps=max_steps,
            stepsize0=1.0e-2,
            stepsize_max=1.0,
            newton_tol=1.0e-10,
        )
        # pacopy.branch_switching(
        #     problem,
        #     u0,
        #     mu0,
        #     callback,
        #     max_steps=max_steps,
        #     stepsize0=1.0e-2,
        #     stepsize_max=1.0,
        #     newton_tol=1.0e-10,
        # )


if __name__ == "__main__":
    # test_self_adjointness()
    # test_f()
    # test_df_dlmbda()
    # test_jacobian()
    test_continuation(max_steps=100)
