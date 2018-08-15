# -*- coding: utf-8 -*-
#
from .newton import newton, NewtonConvergenceError


def natural(
    problem,
    u0,
    lambda0,
    callback,
    stepsize=1.0e-1,
    max_steps=100,
    verbose=True,
    first_order_predictor=False,
):
    """Natural parameter continuation
    """
    lmbda = lambda0

    k = 0
    while True:
        try:
            u = newton(
                lambda u: problem.f(u, lmbda),
                lambda u, rhs: problem.jacobian_solver(u, lmbda, rhs),
                u0,
                tol=1.0e-12,
            )
        except NewtonConvergenceError:
            print("No convergence for lambda={}.".format(lmbda))
            break

        callback(k, lmbda, u)

        k += 1
        if k > max_steps:
            break

        lmbda += stepsize
        if first_order_predictor:
            u0 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
        else:
            u0 = u

    return
