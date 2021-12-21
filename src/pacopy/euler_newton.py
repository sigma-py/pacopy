from __future__ import annotations

import math
from typing import Callable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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
    raise NotImplementedError()


def euler_newton(
    problem,
    u0,
    lmbda0: float,
    callback: Callable,
    max_steps: float = float("inf"),
    max_num_retries: float = float("inf"),
    verbose: bool = True,
    newton_tol: float = 1.0e-12,
    max_newton_steps: int = 5,
    predictor_variant: Literal["tangent"] | Literal["secant"] = "tangent",
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
        stepsize0 (float): Initial step size
        stepsize_max (float): Maximum step size
        stepsize_aggressiveness (float): The step size is adapted after each
            step such that :code:`max_newton_steps` is exhausted approximately.
            This parameter determines how aggressively the step size is increased
            if too few Newton steps were used.
        cos_alpha_min (float): To make a plotted parameter-solution norm curve
            look smooth, subsequent predictors should have a small angle between
            them. The correct way to do this would be to look at

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
        prev_eigval, _ = problem.jacobian_eigenvalue(u, lmbda)

    ds = abs(stepsize0)

    theta2 = theta0 ** 2

    # tangent predictor for the first step
    du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
    dlmbda_ds = 1.0 if stepsize0 > 0 else -1.0
    du_ds = du_dlmbda * dlmbda_ds

    duds2 = problem.inner(du_ds, du_ds)

    # theta2 = (dlmbda_ds ** 2 / dlmbda_ds_target2) * (1 - dlmbda_ds_target2) / duds2
    # theta2 = min(theta2, theta2_max)
    # theta2 = 1.0  # TODO remove
    nrm = math.sqrt(theta2 * duds2 + dlmbda_ds ** 2)
    du_ds /= nrm
    dlmbda_ds /= nrm
    duds2 /= nrm ** 2

    u_current = u
    lmbda_current = lmbda
    du_ds_current = du_ds
    dlmbda_ds_current = dlmbda_ds
    duds2_current = duds2

    num_retries = 0
    callback(k, lmbda, u)
    k += 1

    while True:
        if k > max_steps:
            break

        if num_retries > max_num_retries:
            console.print(
                "[red]"
                f"Maximum number of retries reached ({max_num_retries}) in step {k}. "
                + "Abort."
                + "[/red]",
                highlight=False,
            )
            break

        if verbose:
            uu = problem.inner(u_current, u_current)
            string = f" (retry #{num_retries})" if num_retries > 0 else ""
            console.print(
                "\n[blue]"
                + f"[bold]Step {k}[/bold]{string}\n"
                + f"lmbda = {lmbda_current:.3f}, <u, u> = {uu:.3f}, stepsize = {ds:.3e}"
                + "[/blue]",
                highlight=False,
            )

        # Predictor
        u = u_current + du_ds_current * ds
        lmbda = lmbda_current + dlmbda_ds_current * ds

        if verbose:
            print("Corrector >")
        # Newton corrector
        try:
            u, lmbda, num_newton_steps, newton_success = _newton_corrector(
                problem,
                u,
                lmbda,
                theta2,
                u_current,
                lmbda_current,
                ds,
                max_newton_steps,
                newton_tol,
                verbose,
            )
        except JacobianSolverError:
            if verbose:
                console.print(
                    "[red]Jacobian solver error![/]\n"
                    + "[yellow]Restarting with smaller stepsize.[/]"
                )
            ds *= 0.5
            num_retries += 1
            continue

        if not newton_success:
            if verbose:
                console.print(
                    "[yellow]Newton convergence failure!"
                    + "\nRestarting with smaller step size.[/]"
                )
            ds *= 0.5
            num_retries += 1
            continue

        if converge_onto_zero_eigenvalue:
            eigval, eigvec = problem.jacobian_eigenvalue(u, lmbda)
            is_zero = abs(eigval) < tol

            if is_zero:
                if verbose:
                    console.print("[green]Converged onto zero eigenvalue.[/]")
                return eigval, eigvec
            else:
                # Check if the eigenvalue crossed the origin
                if (prev_eigval > 0 and eigval > 0) or (prev_eigval < 0 and eigval < 0):
                    prev_eigval = eigval
                else:
                    # crossed the origin!
                    if verbose:
                        console.print(
                            "[yellow]Eigenvalue crossed origin!"
                            + "\nRestarting with smaller step size.[/]"
                        )
                    # order 1 approximation for the zero eigenvalue
                    ds *= prev_eigval / (prev_eigval - eigval)
                    continue

        if verbose:
            print("Next predictor >")

        # Approximate dlmbda/ds and du/ds for the next predictor step. Do that
        # here so we can abort the step if the angle between this and previous
        # predictor is too large.
        if predictor_variant == "tangent":
            # tangent predictor (like in natural continuation)
            #
            try:
                du_dlmbda = problem.jacobian_solver(
                    u, lmbda, -problem.df_dlmbda(u, lmbda)
                )
            except JacobianSolverError:
                if verbose:
                    console.print(
                        "[red]Jacobian solver error in tangent predictor! Abort.[/]"
                    )
                raise
            # Make sure the sign of dlambda_ds is correct
            r = theta2 * problem.inner(du_dlmbda, u - u_current) + (
                lmbda - lmbda_current
            )
            dlmbda_ds = 1.0 if r > 0 else -1.0
            du_ds = du_dlmbda * dlmbda_ds
        else:
            # secant predictor
            assert predictor_variant == "secant"
            du_ds = (u - u_current) / ds
            dlmbda_ds = (lmbda - lmbda_current) / ds
            # du_dlmbda not necessary here.
            # du_dlmbda = du_ds / dlmbda_ds

        # At this point, du_ds and dlmbda_ds are still unscaled so they do NOT
        # correspond to the true du/ds and dlmbda/ds yet.

        # To make a plotted parameter-solution norm curve look smooth,
        # subsequent predictors should have a small angle between them. The
        # perfect way of doing this would be to look at
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
            if verbose:
                console.print(
                    "[yellow]Angle between subsequent predictors too large "
                    + f"(cos(alpha) = {cos_alpha:.3f} < {cos_alpha_min:.3f})."
                    + "\nRestarting with smaller step size.[/]",
                    highlight=False,
                )
            ds *= 0.5
            num_retries += 1
            continue

        nrm = math.sqrt(theta2 * duds2 + dlmbda_ds ** 2)
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
            theta2 *= (
                dlmbda_ds ** 2
                / dlmbda_ds2_target
                * (1 - dlmbda_ds2_target)
                / (1 - dlmbda_ds ** 2)
            )
            theta2 = min(1.0e2, theta2)
            theta2 = max(1.0e-2, theta2)

        callback(k, lmbda, u)
        k += 1
        num_retries = 0

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
    theta2: float,
    u_current,
    lmbda_current: float,
    ds: float,
    max_newton_steps: int,
    newton_tol: float,
    verbose: bool,
):
    """Solve the nonlinear equations

       f(u, lmbda) = 0
       theta ** 2 * <du_ds_1, du_ds_1> + dlmbda_ds_1 ** 2 = 1.0,

       du_ds_1  = (u - u_current) / ds
       dlmbda_ds_1  = (lmbda - lmbda_current) / ds

    The second equation is the relative error of the target step size `ds`.
    """
    console = Console()

    # Newton corrector
    num_newton_steps = 0
    newton_success = False

    while True:
        print()
        print("u", problem.inner(u, u))
        print("lmbda", lmbda)
        r = problem.f(u, lmbda)
        du_ds_1 = (u - u_current) / ds
        dlmbda_ds_1 = (lmbda - lmbda_current) / ds

        rho = theta2 * problem.inner(du_ds_1, du_ds_1) + dlmbda_ds_1 ** 2 - 1.0
        # There also is a "tangent" variant of it with one du_ds_1 replaced by
        # du_ds,
        #
        # rho = theta2 * problem.inner(du_ds, du_ds_1) + dlmbda_ds * dlmbda_ds_1 - 1.0
        #
        # but I can't recall why that was necessary or useful (nschloe, Dec 20,
        # 2021).
        norms2 = (problem.norm2_r(r), rho ** 2)
        if verbose:
            print(
                f"Newton norms: sqrt({norms2[0]:.3e} + {norms2[1]:.3e}) "
                f"= {math.sqrt(norms2[0] + norms2[1]):.3e}"
            )
        if norms2[0] + norms2[1] < newton_tol ** 2:
            if verbose:
                console.print(
                    "[green]"
                    + f"Newton corrector converged after {num_newton_steps} steps."
                    + "[/]"
                )
            newton_success = True
            break

        if num_newton_steps >= max_newton_steps:
            break

        # Solve the Newton update
        #
        #  (J,                     df_dlmbda       ) (du    )  = -(r  )
        #  (theta2 * du_ds * 1/ds, dlmbda_ds * 1/ds) (dlmbda)  = -(rho)
        #
        # (This is for the tangent variant, it's slightly different for secant.)
        #
        # In general form, the equation system
        #
        #  (A   b) (x)     = (r)
        #  (c^T d) (alpha) = (rho)
        #
        # has the solution
        #
        #   z1 = A^{-1} r
        #   z2 = -A^{-1} b
        #   alpha = (rho - c^T z1) / (d + c^T z2)
        #   x = z1 + alpha * z2
        #
        # This respresentation has the advantage that, instead of solving one
        # bordered system, one can solve two systems with A, making use of
        # preconditioner strategies for A etc.
        #
        z1 = problem.jacobian_solver(u, lmbda, -r)
        z2 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))

        dfdl = problem.df_dlmbda(u, lmbda)

        print("df/dl", math.sqrt(problem.inner(dfdl, dfdl)))
        print("f", math.sqrt(problem.inner(r, r)))
        print("z1", math.sqrt(problem.inner(z1, z1)))
        print("z2", math.sqrt(problem.inner(z2, z2)))
        print("ds", ds)

        # secant variant:
        tz1 = theta2 * problem.inner(du_ds_1, z1)
        tz2 = theta2 * problem.inner(du_ds_1, z2)
        # The division by 2 is from the the squared terms in rho
        print("frac1", -rho * ds / 2 - tz1)
        print("frac2", dlmbda_ds_1 + tz2)
        dlmbda = (-rho * ds / 2 - tz1) / (dlmbda_ds_1 + tz2)
        print("l", dlmbda)

        # tangent variant:
        # # dlmbda = alpha
        # # = (-rho - theta2 * <du_ds / ds, z1>)
        # #   / (dlmbda_ds / ds + theta2 * <du_ds / ds, z2>)
        # # = (-rho * ds - theta2 * <du_ds, z1>)
        # #   / (dlmbda_ds + theta2 * <du_ds, z2>)
        # tz1 = theta2 * problem.inner(du_ds, z1)
        # tz2 = theta2 * problem.inner(du_ds, z2)
        # dlmbda = (-rho * ds - tz1) / (dlmbda_ds + tz2)

        # du = z1 + dlmbda * z2
        u += z1 + dlmbda * z2
        lmbda += dlmbda
        num_newton_steps += 1

    du_ds_1 = (u - u_current) / ds
    dlmbda_ds_1 = (lmbda - lmbda_current) / ds

    return u, lmbda, num_newton_steps, newton_success
