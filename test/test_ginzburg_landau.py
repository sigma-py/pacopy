import matplotlib
import matplotlib.pyplot as plt
import numpy
import yaml
from scipy.sparse.linalg import spsolve

import cplot
import meshio
import meshplex
import meshzoo
import pacopy
import pyfvm
import pykry


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
        out = (keo * psi) / cv + (self.V + self.g * numpy.abs(psi) ** 2) * psi

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

    def jacobian(self, psi, mu):
        keo = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        cv = self.mesh.control_volumes

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

    def jacobian_solver(self, psi, mu, rhs):
        keo = pyfvm.get_fvm_matrix(self.mesh, edge_kernels=[Energy(mu)])
        cv = self.mesh.control_volumes

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

        jac = self.jacobian(psi, mu)
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

    def jacobian_eigenvalues(self, psi, mu):
        print("a")
        jac = self.jacobian(psi, mu)
        exit(1)
        return


# def test_self_adjointness():
#     points, cells = meshzoo.rectangle(-5.0, 5.0, -5.0, 5.0, 30, 30)
#     mesh = meshplex.MeshTri(points, cells)
#
#     problem = GinzburgLandau(mesh)
#     n = problem.mesh.control_volumes.shape[0]
#     numpy.random.seed(0)
#
#     for _ in range(100):
#         mu = numpy.random.rand(1)
#         psi = numpy.random.rand(n) + 1j * numpy.random.rand(n)
#         jac = problem.jacobian(psi, mu)
#         u = numpy.random.rand(n) + 1j * numpy.random.rand(n)
#         v = numpy.random.rand(n) + 1j * numpy.random.rand(n)
#         a0 = problem.inner(u, jac * v)
#         a1 = problem.inner(jac * u, v)
#         assert abs(a0 - a1) < 1.0e-12
#
#     return


def test_f_i_psi():
    """Assert that <f(psi), i psi> == 0.
    """
    points, cells = meshzoo.rectangle(-5.0, 5.0, -5.0, 5.0, 30, 30)
    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandau(mesh)
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


def test_ginzburg_landau(max_steps=5, n=20):
    a = 10.0
    points, cells = meshzoo.rectangle(-a / 2, a / 2, -a / 2, a / 2, n, n)
    mesh = meshplex.MeshTri(points, cells)

    problem = GinzburgLandau(mesh)
    n = problem.mesh.control_volumes.shape[0]
    u0 = numpy.ones(n, dtype=complex)
    mu0 = 0.0

    mu_list = []

    filename = "sol.xdmf"
    writer = meshio.XdmfTimeSeriesWriter(filename)
    writer.write_points_cells(
        problem.mesh.node_coords, {"triangle": problem.mesh.cells["nodes"]}
    )

    def callback(k, mu, sol):
        mu_list.append(mu)
        # Store the solution
        psi = numpy.array([numpy.real(sol), numpy.imag(sol)]).T
        writer.write_data(k, point_data={"psi": psi})
        with open("data.yml", "w") as fh:
            yaml.dump({"filename": filename, "mu": [float(m) for m in mu_list]}, fh)
        return

    # pacopy.natural(
    #     problem,
    #     u0,
    #     mu0,
    #     callback,
    #     max_steps=1000,
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
    return


def gibbs_energy(mesh, psi):
    """Compute the Gibbs free energy. Useful for plotting purposes.
    """
    psi2 = psi ** 2
    alpha = -numpy.real(numpy.dot(psi2.conj(), mesh.control_volumes * psi2))
    return alpha / numpy.sum(mesh.control_volumes)


def plot_data():
    filename = "data.yml"
    with open(filename, "r") as fh:
        data = yaml.safe_load(fh)

    reader = meshio.XdmfTimeSeriesReader(data["filename"])
    points, cells = reader.read_points_cells()
    x, y, _ = points.T

    # compute all energies in advance
    energies = []
    mesh = meshplex.MeshTri(points, cells["triangle"])
    for k in range(len(data["mu"])):
        _, point_data, _ = reader.read_data(k)
        psi = point_data["psi"]
        psi = psi[:, 0] + 1j * psi[:, 1]
        energies.append(gibbs_energy(mesh, psi))
    energies = numpy.array(energies)

    for k in range(len(data["mu"])):
        plt.figure(figsize=(11, 4))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(data["mu"], energies)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(-1.0, 0.0)
        ax1.grid()
        ax1.plot(data["mu"][k], energies[k], "o", color="#1f77f4")
        ax1.set_xlabel("$\\mu$")
        ax1.set_ylabel("$\\mathcal{E}(\\psi)$")

        _, point_data, _ = reader.read_data(k)
        psi = point_data["psi"]
        psi = psi[:, 0] + 1j * psi[:, 1]

        ax2 = plt.subplot(1, 2, 2)
        triang = matplotlib.tri.Triangulation(x, y)
        # The absolute values of the solution psi of the Ginzburg-Landau equations all
        # sit between 0 and 1, so we don't need a fancy scaling of absolute values for
        # cplot. This results in the values with |psi|=1 being displayed as white,
        # however, losing visual information about the complex argument. On the other
        # hand, plots are rather more bright, resulting in more visually appealing
        # figures.
        cplot.tripcolor(triang, psi, abs_scaling=lambda r: r)
        # plt.tripcolor(triang, numpy.abs(psi))

        ax2.axis("square")
        ax2.set_xlim(-5.0, 5.0)
        ax2.set_ylim(-5.0, 5.0)
        ax2.set_title("$\\psi$")
        # plt.colorbar()
        # plt.set_cmap("gray")
        # plt.clim(0.0, 1.0)

        plt.tight_layout()
        plt.savefig("fig{:03d}.png".format(k))
        # plt.show()
        plt.close()

    return


if __name__ == "__main__":
    # test_self_adjointness()
    # test_f_i_psi()
    # test_df_dlmbda()
    test_ginzburg_landau(max_steps=100, n=100)
    # plot_data()
