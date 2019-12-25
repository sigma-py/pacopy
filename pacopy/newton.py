import math


class NewtonConvergenceError(Exception):
    pass


def newton(f, jacobian_solver, norm2, u0, tol=1.0e-10, max_iter=20, verbose=True):
    u = u0

    fu = f(u)
    nrm = math.sqrt(norm2(fu))
    if verbose:
        print(f"||F(u)|| = {nrm:e}")

    k = 0
    while k < max_iter:
        if nrm < tol:
            break
        du = jacobian_solver(u, -fu)
        u += du
        fu = f(u)
        nrm = math.sqrt(norm2(fu))
        k += 1
        if verbose:
            print(f"||F(u)|| = {nrm:e}")

    is_converged = nrm < tol

    if not is_converged:
        raise NewtonConvergenceError(
            f"Newton's method didn't converge after {k} steps."
        )

    return u, k
