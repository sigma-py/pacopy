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
    newton_tol=1.0e-14,
    newton_max_steps=5,
    predictor="tangent",
    corrector_variant="tangent",
    #
    stepsize0=1.0e0,
    stepsize_max=1.0e0,
    stepsize_aggressiveness=2,
    smoothness_factor=1.0,
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

    ds = stepsize0
    u_prev = None
    lmbda_prev = None
    u_current = u.copy()
    lmbda_current = lmbda

    # threshold after which theta is rescaled
    lmbda_prime_max = 0.0
    # square of the target for dlmbda_ds
    dlmbda_ds_target2 = 0.5
    theta = 1.0

    # tangent predictor for the first step
    du_dlmbda = problem.jacobian_solver(
        u_current, lmbda_current, -problem.df_dlmbda(u_current, lmbda_current)
    )
    # One could optionally use a negative sign here
    dlmbda_ds = 1.0
    du_ds = du_dlmbda * dlmbda_ds
    nrm = math.sqrt(theta ** 2 * problem.inner(du_ds, du_ds) + dlmbda_ds ** 2)
    du_ds /= nrm
    dlmbda_ds /= nrm

    du_ds_norm = math.sqrt(problem.inner(du_ds, du_ds))

    callback(k, lmbda, u, lmbda, u, du_dlmbda)
    k += 1

    while True:
        if k > max_steps:
            break

        print(
            "Compare: {:.3e}  {:.3e}  {:.3e}".format(
                dlmbda_ds ** 2,
                theta ** 2 * problem.inner(du_ds, du_ds),
                dlmbda_ds ** 2 + theta ** 2 * problem.inner(du_ds, du_ds),
            )
        )

        if verbose:
            print()
            print("Step {}, stepsize: {:.3e}".format(k, ds))

        # Predictor
        u = u_current + du_ds * ds
        lmbda = lmbda_current + dlmbda_ds * ds

        # for debugging
        u_predictor = u.copy()
        lmbda_predictor = lmbda

        # Newton corrector
        u, lmbda, num_newton_steps, newton_success = _newton_corrector(
            problem,
            u,
            lmbda,
            theta,
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
            ds = min(stepsize_max, ds)
        else:
            print("Newton convergence failure! Restart with smaller step size.")
            ds *= 0.5
            continue

        # Possibly rescale theta; see LOCA book
        # <http://www.cs.sandia.gov/loca/loca1.1_book.pdf>
        if dlmbda_ds ** 2 > lmbda_prime_max ** 2:
            if abs(dlmbda_ds ** 2 - 1.0) < 1.0e-15:
                theta = 1.0e8
            else:
                # LOCA book, eq. (2.23)
                theta *= math.sqrt(
                    dlmbda_ds ** 2
                    / (1 - dlmbda_ds ** 2)
                    * (1 - dlmbda_ds_target2)
                    / dlmbda_ds_target2
                )
                theta = min(theta, 1.0e8)

        callback(k, lmbda, u, lmbda_predictor, u_predictor, du_dlmbda)
        k += 1
        u_prev = u_current
        lmbda_prev = lmbda_current
        u_current = u
        lmbda_current = lmbda

        du_ds_prev = du_ds.copy()

        # Approximate dlmbda/ds and du/ds for the next predictor step
        if predictor == "tangent":
            # tangent predictor (like in natural continuation)
            #
            du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
            # Make sure the sign of dlambda_ds is correct
            r = theta ** 2 * problem.inner(du_dlmbda, u - u_prev) + (lmbda - lmbda_prev)
            dlmbda_ds = 1.0 if r > 0 else -1.0
            du_ds = du_dlmbda * dlmbda_ds
        else:
            # secant predictor
            assert predictor == "secant"
            du_ds = (u_current - u_prev) / ds
            dlmbda_ds = (lmbda_current - lmbda_prev) / ds
        nrm = math.sqrt(theta ** 2 * problem.inner(du_ds, du_ds) + dlmbda_ds ** 2)
        du_ds /= nrm
        dlmbda_ds /= nrm

        du_ds_norm_prev = du_ds_norm
        du_ds_norm = math.sqrt(problem.inner(du_ds, du_ds))

        # LOCA book (2.25), with du/ds swapped in for du/dlmbda to make it compatible
        # with the secant predictor. (Identical to LOCA if one uses tangent.)
        # For most test problems, this factor seems to be very close to 1, so it doesn't
        # really do anything.
        # TODO find out if this is useful at all
        tangent_factor = (
            problem.inner(du_ds, du_ds_prev)
            / du_ds_norm
            / du_ds_norm_prev
        )
        assert tangent_factor > 0
        ds *= tangent_factor ** smoothness_factor

    return None


def _newton_corrector(
    problem,
    u,
    lmbda,
    theta,
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
        r = problem.f(u, lmbda)
        if corrector_variant == "tangent":
            q = (
                theta ** 2 * problem.inner(u - u_current, du_ds)
                + (lmbda - lmbda_current) * dlmbda_ds
                - ds
            )
        else:
            assert corrector_variant == "secant"
            q = (
                theta ** 2 * problem.inner(u - u_current, u - u_current)
                + (lmbda - lmbda_current) ** 2
                - ds ** 2
            )

        print(
            "Newton norms: sqrt({:.3e} + {:.3e}) = {:.3e}".format(
                problem.norm2_r(r), q ** 2, math.sqrt(problem.norm2_r(r) + q ** 2)
            )
        )
        if problem.norm2_r(r) + q ** 2 < newton_tol ** 2:
            print("Newton corrector converged after {} steps.".format(num_newton_steps))
            newton_success = True
            break

        if num_newton_steps >= newton_max_steps:
            break

        z1 = problem.jacobian_solver(u, lmbda, -r)
        z2 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))

        if corrector_variant == "tangent":
            dlmbda = -(q + theta ** 2 * problem.inner(du_ds, z1)) / (
                dlmbda_ds + theta ** 2 * problem.inner(du_ds, z2)
            )
        else:
            assert corrector_variant == "secant"
            dlmbda = -(q + 2 * theta ** 2 * problem.inner(u - u_current, z1)) / (
                2 * (lmbda - lmbda_current)
                + 2 * theta ** 2 * problem.inner(u - u_current, z2)
            )

        du = z1 + dlmbda * z2

        u += du
        lmbda += dlmbda
        num_newton_steps += 1

    return u, lmbda, num_newton_steps, newton_success
