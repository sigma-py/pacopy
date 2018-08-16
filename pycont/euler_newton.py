# -*- coding: utf-8 -*-
#
import numpy

from .newton import newton, NewtonConvergenceError


def tangent(u, lmbda):
    """Computes the normalized arc tangent (du/ds, lmbda/ds). Computation is based on
    the equations

       ||du/ds||^2 + (dlmbda/ds)^2 = 1,
       d/ds f(u(s), lmbda(s)) = 0.

    They can be reduced to the nonlinear equation

        0 = df/du v + df/dlmbda sqrt(1 - ||v||^2)

    for `v := du/ds`. Its Jacobian is

        J(v) = df/du - 2 df/dlmbda v^T / sqrt(1 - ||v||^2),

    a rank-1 update to df/du which can be solved via Sherman-Morrison.

    Note that this does not work at turning points where `dlmbda/ds = 0`. Here, `v` is
    the nontrivial solution to

       0 = df/du v.
    """
    return


def euler_newton(
    problem,
    u0,
    lmbda0,
    callback,
    stepsize=1.0e-1,
    max_steps=100,
    verbose=True,
    newton_tol=1.0e-12,
    newton_max_steps=5,
):
    """Pseudo-arclength continuation, implemented in the style of LOCA
    <https://trilinos.org/packages/nox-and-loca/>, i.e., one doesn't solve a bordered
    system, but the solution is constructed from two solves of the naked Jacobian
    system. This has several advantages, one being that preconditioners for the Jacobian
    can be reused.
    """
    lmbda = lmbda0

    k = 0
    try:
        u, _ = newton(
            lambda u: problem.f(u, lmbda),
            lambda u, rhs: problem.jacobian_solver(u, lmbda, rhs),
            u0,
            tol=newton_tol,
            max_iter=newton_max_steps,
        )
    except NewtonConvergenceError as e:
        print("No convergence for initial step.".format(lmbda))
        raise e

    callback(k, lmbda, u)
    k += 1

    delta_s = stepsize

    u_prev = None
    lmbda_prev = None
    u_current = u.copy()
    lmbda_current = lmbda

    # TODO replace dot product by problem.inner?
    while True:
        if k > max_steps:
            break

        # Predictor
        # Use the secant predictor if possible. In the first step, this is not possible
        # and we're doing something else.
        if k == 1:
            du_dlmbda = problem.jacobian_solver(
                u_current, lmbda, -problem.df_dlmbda(u_current, lmbda_current)
            )
            dlmbda_ds = 1 / numpy.sqrt(1 + numpy.dot(du_dlmbda, du_dlmbda))
            du_ds = du_dlmbda * dlmbda_ds
            # du_ds, dlmbda_ds are chosen normalized.
        else:
            # secant predictor
            du_ds = (u_current - u_prev) / delta_s
            dlmbda_ds = (lmbda_current - lmbda_prev) / delta_s
            tangent_length = numpy.sqrt(numpy.dot(du_ds, du_ds) + dlmbda_ds ** 2)
            du_ds /= tangent_length
            dlmbda_ds /= tangent_length

        u = u_current + du_ds * delta_s
        lmbda = lmbda_current + dlmbda_ds * delta_s

        if verbose:
            print("Step {} (predictor): lambda  {:.3e}".format(k, lmbda))

        # Newton corrector
        num_newton_steps = 0
        while True:
            r = problem.f(u, lmbda)
            q = (
                numpy.dot(u - u_current, u - u_current)
                + (lmbda - lmbda_current) ** 2
                - delta_s ** 2
            )

            if numpy.dot(r, r) + q ** 2 < newton_tol:
                print(
                    "Newton corrector converged after {} steps.".format(
                        num_newton_steps
                    )
                )
                break

            z1 = problem.jacobian_solver(u, lmbda, -r)
            z2 = problem.jacobian_solver(u, lmbda, problem.df_dlmbda(u, lmbda))

            dlmbda = (-q - 2 * numpy.dot(u - u_current, z1)) / (
                2 * (lmbda - lmbda_current) - 2 * numpy.dot(u - u_current, z2)
            )
            du = z1 - dlmbda * z2

            u += du
            lmbda += dlmbda
            num_newton_steps += 1

        if verbose:
            print("Step {} (final): lambda  {:.3e}\n".format(k, lmbda))

        callback(k, lmbda, u)
        k += 1
        u_prev = u_current
        lmbda_prev = lmbda_current
        u_current = u
        lmbda_current = lmbda

    return None
