# -*- coding: utf-8 -*-
#
from .newton import newton, NewtonConvergenceError


def natural(
    problem, u0, lambda0, callback, stepsize=1.0e-1, max_steps=100, verbose=True
):
    """Natural parameter continuation
    """
    lmbda = lambda0

    while True:
        try:
            sol = newton(
                lambda u: problem.f(u, lmbda),
                lambda u, rhs: problem.jacobian_solver(u, lmbda, rhs),
                u0,
                tol=1.0e-12,
            )
        except NewtonConvergenceError:
            print("No convergence for lambda={}.".format(lmbda))
            break

        callback(lmbda, sol)
        lmbda += stepsize

    exit(1)
    return None
