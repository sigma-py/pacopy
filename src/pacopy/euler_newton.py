from __future__ import annotations

import math
from typing import Callable

from rich.console import Console

from .errors import JacobianSolverError
from .newton import NewtonConvergenceError, newton


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
    pass


def euler_newton(
    problem,
    u0,
    lmbda0: float,
    callback: Callable,
    max_steps: float = float("inf"),
    verbose: bool = True,
    newton_tol: float = 1.0e-12,
    max_newton_steps: int = 5,
    predictor_variant: str = "tangent",
    corrector_variant: str = "tangent",
    # predictor_variant: Literal["tangent"] | Literal["secant"] = "tangent",
    # corrector_variant: Literal["tangent"] | Literal["secant"] = "tangent",
    #
    stepsize0: float = 5.0e-1,
    stepsize_max: float = float("inf"),
    stepsize_aggressiveness: int = 2,
    cos_alpha_min: float = 0.9,
    theta0: float = 1.0,
    adaptive_theta: bool = False,
    converge_onto_zero_eigenvalue: bool = False,
):
    """Pseudo-arclength continuation.

    This implementation takes some inspiration from `LOCA
    <http://www.cs.sandia.gov/loca/loca1.1_book.pdf>`_ in that no single bordered system
    is solved, but the solution is constructed from two solves of the plain Jacobian
    system. This has several advantages, one being that preconditioners for the Jacobian
    can be reused.

    Args:
        problem: Instance of the problem class
        u0: Initial guess
        lambda0: Initial parameter value
        callback: Callback function
        max_steps: Maximum number of continuation steps
        verbose: Verbose output
        newton_tol: Newton tolerance
        max_newton_steps: Maxmimum number of Newton steps
        predictor_variant (string): :code:`"tangent"` or :code:`"secant"`
        corrector_variant (string): :code:`"tangent"` or :code:`"secant"`
        stepsize0 (float): Initial step size
        stepsize_max (float): Maximum step size
        stepsize_aggressiveness (float): The step size is adapted after each step
            such that :code:`max_newton_steps` is exhausted approximately. This
            parameter determines how aggressively the the step size is increased if too
            few Newton steps were used.
        cos_alpha_min (float): To make a plotted parameter-solution norm curve look
            smooth, subsequent predictors should have a small angle between them. The
            correct way to do this would be to look at

            .. math::

                  \\cos(\\alpha) = v_i^T v_{i+1} / \\|v_i\\| / \\|v_{i+1}\\| \\\\
                  v_i := (\\|u_i^p\\| - \\|u_i\\|, \\lambda_i^p - \\lambda_i).

            Instead of taking the solution and predictor norms, the entire vectors are
            taken,

            .. math::

              w_i := (u_i^p - u_i, \\lambda_i^p - \\lambda_i),

            with the appropriate inner product in the product space :math:`U\\times
            \\mathbb{R}`. This results in the expression

            .. math::

                \\cos(\\alpha) = \\frac{
                    \\langle \\frac{du_0}{ds}, \\frac{du}{ds}\\rangle +
                    \\frac{d\\lambda_0}{ds} \\frac{d\\lambda}{ds}}{
                    \\sqrt{\\|\\frac{du_0}{ds}\\|^2
                    + \\frac{d\\lambda_0}{ds}^2}
                    \\sqrt{\\|\\frac{du}{ds}\\|^2 +
                    \\frac{d\\lambda}{ds}^2}}

            When using the tangent predictor, this can be written as

            .. math::

              \\cos(\\alpha) = \\frac{\\langle \\frac{du}{d\\lambda},
              \\frac{du_0}{d\\lambda}\\rangle + 1}{\\sqrt{\\|\\frac{du}{d\\lambda}\\|^2 + 1}
                  \\sqrt{\\|\\frac{du_0}{d\\lambda}\\|^2 + 1}
              }

            When removing the :math:`+1` s, this is the expression that is used in LOCA
            (equation (2.25) in the LOCA book).

        theta0 (float): The arc-length equation is

            .. math::

              \\left\\|\\frac{du}{ds}\\right\\|^2 +
              \\left(\\frac{d\\lambda}{ds}\\right)^2 = 1.

            Quoting from LOCA:
            It is numerically advantageous for the relative magnitudes of the parameter
            and solution updates to be of similar order. In particular, the advantage of
            using arc-length parameterization can be lost if the solution contribution
            to the arc length equation becomes very small. In this algorithm, a single
            scaling factor :math:`\\theta` is used for the solution contribution in
            order to provide some control over the relative contributions of
            :math:`\\lambda` and :math:`u`. The modified arc length equation is then

            .. math::

              \\left\\|\\frac{du}{ds}\\right\\|^2 +
              \\theta^2 \\left(\\frac{d\\lambda}{ds}\\right)^2 = 1.
    """
    lmbda = lmbda0

    console = Console()

    k = 0
    try:
        u, _ = newton(
            lambda u: problem.f(u, lmbda),
            lambda u, rhs: problem.jacobian_solver(u, lmbda, rhs),
            problem.norm2_r,
            u0,
            tol=newton_tol,
            max_iter=max_newton_steps,
        )
    except NewtonConvergenceError as e:
        console.print("[red]No convergence for initial step.[/]")
        raise e

    if converge_onto_zero_eigenvalue:
        # Track _one_ nonzero eigenvalue.
        tol = 1.0e-10
        nonzero_eigval, _ = problem.jacobian_eigenvalue(u, lmbda)

    ds = abs(stepsize0)

    theta = theta0

    # tangent predictor for the first step
    du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
    dlmbda_ds = 1.0 if stepsize0 > 0 else -1.0
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
            console.print(
                f"\n[blue][bold]Step {k}[/bold], stepsize: {ds:.3e}[blue]",
                highlight=False,
            )

        # Predictor
        u = u_current + du_ds_current * ds
        lmbda = lmbda_current + dlmbda_ds_current * ds

        # Newton corrector
        try:
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
                max_newton_steps,
                newton_tol,
            )
        except JacobianSolverError:
            console.print(
                "[red]Jacobian solver error!\nRestarting with smaller stepsize.[/]"
            )
            ds *= 0.5
            continue

        if not newton_success:
            console.print(
                "[yellow]Newton convergence failure!"
                + "\nRestarting with smaller step size.[/]"
            )
            ds *= 0.5
            continue

        if converge_onto_zero_eigenvalue:
            eigval, eigvec = problem.jacobian_eigenvalue(u, lmbda)
            is_zero = abs(eigval) < tol

            if is_zero:
                console.print("[green]Converged onto zero eigenvalue.[/]")
                return eigval, eigvec
            else:
                # Check if the eigenvalue crossed the origin
                if (nonzero_eigval > 0 and eigval > 0) or (
                    nonzero_eigval < 0 and eigval < 0
                ):
                    nonzero_eigval = eigval
                else:
                    # crossed the origin!
                    console.print(
                        "[yellow]Eigenvalue crossed origin!"
                        + "\nRestarting with smaller step size.[/]"
                    )
                    # order 1 approximation for the zero eigenvalue
                    ds *= nonzero_eigval / (nonzero_eigval - eigval)
                    continue

        # Approximate dlmbda/ds and du/ds for the next predictor step
        if predictor_variant == "tangent":
            # tangent predictor (like in natural continuation)
            #
            try:
                du_dlmbda = problem.jacobian_solver(
                    u, lmbda, -problem.df_dlmbda(u, lmbda)
                )
            except JacobianSolverError:
                console.print(
                    "[red]Jacobian solver error in tangent predictor! Abort.[/]"
                )
                raise
            # Make sure the sign of dlambda_ds is correct
            r = theta ** 2 * problem.inner(du_dlmbda, u - u_current) + (
                lmbda - lmbda_current
            )
            dlmbda_ds = 1.0 if r > 0 else -1.0
            du_ds = du_dlmbda * dlmbda_ds
        else:
            # secant predictor
            assert predictor_variant == "secant"
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
            console.print(
                "[yellow]Angle between subsequent predictors too large "
                + f"(cos(alpha) = {cos_alpha:.3f} < {cos_alpha_min:.3f})."
                + "\nRestarting with smaller step size.[/]",
                highlight=False,
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
            * ((max_newton_steps - num_newton_steps) / (max_newton_steps - 1)) ** 2
        )
        ds = min(stepsize_max, ds)


def _newton_corrector(
    problem,
    u,
    lmbda: float,
    theta,
    u_current,
    lmbda_current: float,
    du_ds,
    dlmbda_ds: float,
    ds,
    corrector_variant: str,
    max_newton_steps: int,
    newton_tol: float,
):
    console = Console()

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

        norms2 = (problem.norm2_r(r), q ** 2)
        print(
            f"Newton norms: sqrt({norms2[0]:.3e} + {norms2[1]:.3e}) "
            f"= {math.sqrt(norms2[0] + norms2[1]):.3e}"
        )
        if norms2[0] + norms2[1] < newton_tol ** 2:
            console.print(
                f"[green]Newton corrector converged after {num_newton_steps} steps.[/]"
            )
            print(f"lmbda = {lmbda}, <u, u> = {problem.inner(u, u)}")
            newton_success = True
            break

        if num_newton_steps >= max_newton_steps:
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
