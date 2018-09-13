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
    max_steps=float("inf"),
    verbose=True,
    newton_tol=1.0e-12,
    newton_max_steps=5,
    predictor="tangent",
    corrector_variant="tangent",
    #
    stepsize0=5.0e-1,
    stepsize_max=float("inf"),
    stepsize_aggressiveness=2,
    cos_alpha_min=0.9,
    theta0=1.0,
    adaptive_theta=False,
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

    theta = theta0

    # tangent predictor for the first step
    du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
    # One could optionally use a negative sign here
    dlmbda_ds = 1.0
    du_ds = du_dlmbda * dlmbda_ds

    duds2 = problem.inner(du_ds, du_ds)

    # theta = math.sqrt(
    #     (dlmbda_ds ** 2 / dlmbda_ds_target2) * (1 - dlmbda_ds_target2) / duds2
    # )
    # theta = min(theta, theta_max)
    # theta = 1.0  # TODO remove
    nrm = math.sqrt(theta ** 2 * duds2 + dlmbda_ds ** 2)
    du_ds /= nrm
    dlmbda_ds /= nrm
    duds2 /= nrm ** 2

    u_current = u
    lmbda_current = lmbda
    du_ds_current = du_ds
    dlmbda_ds_current = dlmbda_ds
    duds2_current = duds2

    callback(k, lmbda, u)
    k += 1

    while True:
        if k > max_steps:
            break

        if verbose:
            print()
            print("Step {}, stepsize: {:.3e}".format(k, ds))

        # Predictor
        u = u_current + du_ds_current * ds
        lmbda = lmbda_current + dlmbda_ds_current * ds

        # Newton corrector
        u, lmbda, num_newton_steps, newton_success = _newton_corrector(
            problem,
            u,
            lmbda,
            theta,
            u_current,
            lmbda_current,
            du_ds_current,
            dlmbda_ds_current,
            ds,
            corrector_variant,
            newton_max_steps,
            newton_tol,
        )

        if not newton_success:
            print("Newton convergence failure! Restart with smaller step size.")
            ds *= 0.5
            continue

        # Approximate dlmbda/ds and du/ds for the next predictor step
        if predictor == "tangent":
            # tangent predictor (like in natural continuation)
            #
            du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
            # Make sure the sign of dlambda_ds is correct
            r = theta ** 2 * problem.inner(du_dlmbda, u - u_current) + (
                lmbda - lmbda_current
            )
            dlmbda_ds = 1.0 if r > 0 else -1.0
            du_ds = du_dlmbda * dlmbda_ds
        else:
            # secant predictor
            assert predictor == "secant"
            du_ds = (u - u_current) / ds
            dlmbda_ds = (lmbda - lmbda_current) / ds
            # du_lmbda not necessary here. TODO remove
            du_dlmbda = du_ds / dlmbda_ds

        # At this point, du_ds and dlmbda_ds are still unscaled so they do NOT
        # correspond to the true du/ds and dlmbda/ds yet.

        # To make a plotted parameter-solution norm curve look smooth, subsequent
        # predictors should have a small angle between them. The correct way to do this
        # would be to look at
        #
        #   cos(alpha) = v_i^T v_{i+1} / ||v_i|| / ||v_{i+1}||
        #   v_i := (||u_i^p|| - ||u_i||, lmbda_i^p - lmbda_i).
        #
        # Instead of taking the solution and predictor norms, the entire vectors are
        # taken,
        #
        #   w_i := (u_i^p - u_i, lmbda_i^p - lmbda_i),
        #
        # with the appropriate inner product in the product space (u)x(lmbda). This
        # results in the expression
        duds2 = problem.inner(du_ds, du_ds)
        cos_alpha = (
            (problem.inner(du_ds_current, du_ds) + (dlmbda_ds_current * dlmbda_ds))
            / math.sqrt(duds2_current + dlmbda_ds_current ** 2)
            / math.sqrt(duds2 + dlmbda_ds ** 2)
        )
        # When using the tangent predictor, this can be written as
        #
        #   cos_alpha = (
        #       (problem.inner(du_dlmbda, du_dlmbda_prev) + 1)
        #       / math.sqrt(problem.inner(du_dlmbda, du_dlmbda) + 1)
        #       / math.sqrt(problem.inner(du_dlmbda_prev, du_dlmbda_prev) + 1)
        #   )
        #
        # When removing the "+1"s, this is the expression that is used in LOCA (equation
        # (2.25) in the LOCA book).
        if cos_alpha < cos_alpha_min:
            print(
                (
                    "Angle between subsequent predictors too large (cos(alpha) = {} < {}). "
                    "Restart with smaller step size."
                ).format(cos_alpha, cos_alpha_min)
            )
            ds *= 0.5
            continue

        nrm = math.sqrt(theta ** 2 * duds2 + dlmbda_ds ** 2)
        du_ds /= nrm
        dlmbda_ds /= nrm

        u_current = u
        lmbda_current = lmbda
        du_ds_current = du_ds
        # duds2_current could be retrieved by a simple division
        duds2_current = problem.inner(du_ds, du_ds)
        dlmbda_ds_current = dlmbda_ds

        if adaptive_theta:
            # See LOCA manual, equation (2.23). There are min and max safeguards that
            # prevent numerical instabilities when solving the nonlinear systems. Needs
            # more investigation.
            dlmbda_ds2_target = 0.5
            theta *= (
                abs(dlmbda_ds)
                / math.sqrt(dlmbda_ds2_target)
                * math.sqrt((1 - dlmbda_ds2_target) / (1 - dlmbda_ds ** 2))
            )
            theta = min(1.0e1, theta)
            theta = max(1.0e-1, theta)

        callback(k, lmbda, u)
        k += 1

        # Stepsize update
        ds *= (
            1
            + stepsize_aggressiveness
            * ((newton_max_steps - num_newton_steps) / (newton_max_steps - 1)) ** 2
        )
        ds = min(stepsize_max, ds)

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

        # Solve
        #
        #  (J,     dF/dlmbda) (du    )  =  -(F)
        #  (du/ds, dlmbda/ds) (dlmbda)      (q)
        #
        z1 = problem.jacobian_solver(u, lmbda, -r)
        z2 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))

        if corrector_variant == "tangent":
            dlmbda = -(q + theta ** 2 * problem.inner(du_ds, z1)) / (
                dlmbda_ds + theta ** 2 * problem.inner(du_ds, z2)
            )
            du = z1 + dlmbda * z2
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
