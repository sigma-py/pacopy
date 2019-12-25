import math


class NewtonConvergenceError(Exception):
    pass


def newton(f, jacobian_solver, norm2, u0, tol=1.0e-10, max_iter=20, verbose=True):
    u = u0

    fu = f(u)
    nrm = math.sqrt(norm2(fu))
    if verbose:
        print("||F(u)|| = {:e}".format(nrm))

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
            print("||F(u)|| = {:e}".format(nrm))

    is_converged = nrm < tol

    if not is_converged:
        raise NewtonConvergenceError(
            "Newton's method didn't converge after {} steps.".format(k)
        )

    return u, k
