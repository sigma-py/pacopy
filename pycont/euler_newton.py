# -*- coding: utf-8 -*-
#
import math

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
    max_steps=100,
    verbose=True,
    newton_tol=1.0e-11,
    newton_max_steps=5,
    predictor="secant",
    corrector_variant="secant",
    #
    stepsize0=1.0e-1,
    stepsize_min=1.0e-2,
    stepsize_max=1.0e0,
    stepsize_aggressiveness=2,
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
            problem.norm2_r,
            u0,
            tol=newton_tol,
            max_iter=newton_max_steps,
        )
    except NewtonConvergenceError as e:
        print("No convergence for initial step.".format(lmbda))
        raise e

    callback(k, lmbda, u)
    k += 1

    ds = stepsize0

    u_prev = None
    lmbda_prev = None
    u_current = u.copy()
    lmbda_current = lmbda

    # threshold after which theta is rescaled
    lmbda_prime_max = 0.0
    # square of the target for dlmbda_ds
    lmbda_prime_g2 = 0.5
    theta = 1.0

    while True:
        if k > max_steps:
            break

        # Predictor
        if predictor == "tangent" or k == 1:
            # tangent predictor
            # TODO not working at turning points; fix that
            du_dlmbda = problem.jacobian_solver(
                u_current, lmbda_current, -problem.df_dlmbda(u_current, lmbda_current)
            )
            dlmbda_ds = 1 / math.sqrt(
                1 + theta ** 2 * problem.inner(du_dlmbda, du_dlmbda)
            )
            du_ds = du_dlmbda * dlmbda_ds
            # du_ds, dlmbda_ds are chosen normalized.
            if k > 1:
                # Make sure the sign of dlambda_ds is correct
                r = theta ** 2 * problem.inner(du_dlmbda, u_current - u_prev) + (
                    lmbda_current - lmbda_prev
                )
                dlmbda_ds = abs(dlmbda_ds) if r > 0 else -abs(dlmbda_ds)
        else:
            # secant predictor
            # TODO add theta
            assert predictor == "secant"
            du_ds = (u_current - u_prev) / ds
            dlmbda_ds = (lmbda_current - lmbda_prev) / ds
            tangent_length = math.sqrt(problem.inner(du_ds, du_ds) + dlmbda_ds ** 2)
            du_ds /= tangent_length
            dlmbda_ds /= tangent_length

        u = u_current + du_ds * ds
        lmbda = lmbda_current + dlmbda_ds * ds

        if verbose:
            print(
                "Step {} (predictor): stepsize: {:.3e},  lambda  {:.3e} -> {:.3e}".format(
                    k, ds, lmbda_current, lmbda
                )
            )

        # Newton corrector
        u, lmbda, num_newton_steps, newton_success = _newton_corrector(
            problem,
            u,
            lmbda,
            u_current,
            lmbda_current,
            du_ds,
            dlmbda_ds,
            ds,
            corrector_variant,
            newton_max_steps,
            newton_tol,
        )

        if newton_success:
            ds *= (
                1
                + stepsize_aggressiveness
                * ((newton_max_steps - num_newton_steps) / (newton_max_steps - 1)) ** 2
            )
        else:
            print("Newton convergence failure! Restart with smaller step size.")
            ds *= 0.5
            continue

        if verbose:
            print("Step {} (final): lambda  {:.3e}\n".format(k, lmbda))

        # Possibly rescale theta; see LOCA book
        # <http://www.cs.sandia.gov/loca/loca1.1_book.pdf>
        if dlmbda_ds > lmbda_prime_max:
            theta *= (
                dlmbda_ds
                / math.sqrt(lmbda_prime_g2)
                * math.sqrt((1 - lmbda_prime_g2) / (1 - dlmbda_ds ** 2))
            )
            theta = min(theta, 1.0e8)

        callback(k, lmbda, u)
        k += 1
        u_prev = u_current
        lmbda_prev = lmbda_current
        u_current = u
        lmbda_current = lmbda

    return None


def _newton_corrector(
    problem,
    u,
    lmbda,
    u_current,
    lmbda_current,
    du_ds,
    dlmbda_ds,
    ds,
    corrector_variant,
    newton_max_steps,
    newton_tol,
):
    # Newton corrector
    num_newton_steps = 0
    newton_success = False
    while True:
        if num_newton_steps > newton_max_steps:
            break

        r = problem.f(u, lmbda)
        if corrector_variant == "tangent":
            q = (
                problem.inner(u - u_current, du_ds)
                + (lmbda - lmbda_current) * dlmbda_ds
                - ds
            )
        else:
            assert corrector_variant == "secant"
            q = (
                problem.inner(u - u_current, u - u_current)
                + (lmbda - lmbda_current) ** 2
                - ds ** 2
            )

        if problem.norm2_r(r) + q ** 2 < newton_tol ** 2:
            print("Newton corrector converged after {} steps.".format(num_newton_steps))
            newton_success = True
            break

        z1 = problem.jacobian_solver(u, lmbda, -r)
        z2 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))

        if corrector_variant == "tangent":
            dlmbda = -(q + problem.inner(du_ds, z1)) / (
                dlmbda_ds + problem.inner(du_ds, z2)
            )
        else:
            assert corrector_variant == "secant"
            dlmbda = -(q + 2 * problem.inner(u - u_current, z1)) / (
                2 * (lmbda - lmbda_current) + 2 * problem.inner(u - u_current, z2)
            )

        du = z1 + dlmbda * z2

        u += du
        lmbda += dlmbda
        num_newton_steps += 1

    return u, lmbda, num_newton_steps, newton_success
