import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import pacopy


class Bratu1d:
    def __init__(self):
        self.n = 51
        h = 1.0 / (self.n - 1)

        self.H = np.full(self.n, h)
        self.H[0] = h / 2
        self.H[-1] = h / 2

        self.A = (
            scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(self.n, self.n))
            / h ** 2
        )

    def inner(self, a, b):
        return np.dot(a, self.H * b)

    def norm2_r(self, a):
        return np.dot(a, a)

    def f(self, u, lmbda):
        out = self.A.dot(u) - lmbda * np.exp(u)
        out[0] = u[0]
        out[-1] = u[-1]
        return out

    def df_dlmbda(self, u, lmbda):
        out = -np.exp(u)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    def jacobian(self, u, lmbda):
        M = self.A.copy()
        d = M.diagonal().copy()
        d -= lmbda * np.exp(u)
        M.setdiag(d)
        # Dirichlet conditions
        assert np.all(M.offsets == [-1, 0, 1])
        M.data[0][self.n - 2] = 0.0
        M.data[1][0] = 1.0
        M.data[1][self.n - 1] = 1.0
        M.data[2][1] = 0.0
        return M.tocsr()

    def jacobian_solver(self, u, lmbda, rhs):
        return scipy.sparse.linalg.spsolve(self.jacobian(u, lmbda), rhs)


def test_bratu(max_steps=10, update_plot=False):
    problem = Bratu1d()
    u0 = np.zeros(problem.n)
    lmbda0 = 0.0

    # https://stackoverflow.com/a/4098938/353337
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("$||u||_2$")
    ax1.set_xlim(0.0, 4.0)
    ax1.set_ylim(0.0, 6.0)
    ax1.grid()

    (line1,) = ax1.plot([], [], "-x", color="C0")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("solution")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 5.0)
    ax2.grid()

    (line2,) = ax2.plot([], [], "-", color="C0")
    line2.set_xdata(np.linspace(0.0, 1.0, problem.n))

    # line3, = ax2.plot([], [], "-", color="C1")
    # line3.set_xdata(np.linspace(0.0, 1.0, problem.n))

    milestones = np.arange(0.5, 3.2, 0.5)

    def callback(k, lmbda, sol):
        if update_plot:
            line1.set_xdata(np.append(line1.get_xdata(), lmbda))
            # val = np.max(np.abs(sol))
            val = np.sqrt(problem.inner(sol, sol))
            line1.set_ydata(np.append(line1.get_ydata(), val))

            # ax1.plot(
            #     [lmbda_pre], [np.sqrt(problem.inner(u_pre, u_pre))], ".", color="C1"
            # )

            line2.set_ydata(sol)
            # line3.set_ydata(du_dlmbda)

            fig.canvas.draw()
            fig.canvas.flush_events()
            # plt.savefig('bratu1d.png', transparent=True, bbox_inches="tight")

    pacopy.natural(
        problem,
        u0,
        lmbda0,
        callback,
        max_steps=max_steps,
        newton_tol=1e-10,
        milestones=milestones,
    )

    # The condition number of the Jacobian is about 10^4, so we can only expect Newton
    # to converge up to about this factor above machine precision.
    pacopy.euler_newton(
        problem, u0, lmbda0, callback, max_steps=max_steps, newton_tol=1.0e-10
    )


if __name__ == "__main__":
    test_bratu(100, update_plot=True)
    # test_self_adjointness()
