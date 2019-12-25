from .euler_newton import euler_newton


def branch_switching(
    problem,
    u0,
    lmbda0,
    callback,
    max_steps=float("inf"),
    verbose=True,
    newton_tol=1.0e-12,
    max_newton_steps=5,
    predictor_variant="tangent",
    corrector_variant="tangent",
    #
    stepsize0=5.0e-1,
    stepsize_max=float("inf"),
    stepsize_aggressiveness=2,
    cos_alpha_min=0.9,
    theta0=1.0,
    adaptive_theta=False,
):
    eigval, eigvec = euler_newton(
        problem,
        u0,
        lmbda0,
        callback,
        max_steps=max_steps,
        verbose=verbose,
        newton_tol=newton_tol,
        max_newton_steps=max_newton_steps,
        predictor_variant=predictor_variant,
        corrector_variant=corrector_variant,
        #
        stepsize0=stepsize0,
        stepsize_max=stepsize_max,
        stepsize_aggressiveness=stepsize_aggressiveness,
        cos_alpha_min=cos_alpha_min,
        theta0=theta0,
        adaptive_theta=adaptive_theta,
        converge_onto_zero_eigenvalue=True,
    )
    print(eigval)
    print(eigvec)

    import meshio
    import numpy

    filename = "ev.xdmf"
    writer = meshio.xdmf.TimeSeriesWriter(filename)
    writer.write_points_cells(
        problem.mesh.node_coords, {"triangle": problem.mesh.cells["nodes"]}
    )
    psi = numpy.array([eigvec[0::2], eigvec[1::2]]).T
    writer.write_data(0, point_data={"psi": psi})
    exit(1)
