import viennaps as ps


ps.setDimension(3)  # Set to 3 for hole etching in 3D
ps.Logger.setLogLevel(ps.LogLevel.INFO)


def main():

    # Geometry parameters
    grid_delta = 0.05
    x_extent = 3.0
    y_extent = 3.0
    feature_width = 1.0  # Diameter for hole, Width for trench
    mask_height = 1.0
    taper_angle = 0.0

    # Single Particle Process Model
    # Parameters: rate, stickingProbability, yield, maskMaterial
    rate = -2.0
    sticking_probability = 0.2
    source_power = 1.0
    process_time = 1.0

    def run_simulation(temporal_scheme, calc_intermediate):
        suffix = temporal_scheme.name
        if calc_intermediate:
            suffix += "_recalc"

        # Create a Hole in 3D (defaults to trench in 2D)
        domain = ps.Domain(grid_delta, x_extent, y_extent)
        ps.MakeHole(
            domain,
            feature_width / 2.0,
            0.0,
            0.0,
            mask_height,
            taper_angle,
        ).apply()

        model = ps.SingleParticleProcess(
            rate, sticking_probability, source_power, ps.Material.Mask
        )
        process = ps.Process(domain, model, process_time)

        advection_params = ps.AdvectionParameters()
        advection_params.spatialScheme = ps.SpatialScheme.WENO_5TH_ORDER
        advection_params.temporalScheme = temporal_scheme
        advection_params.calculateIntermediateVelocities = calc_intermediate
        process.setParameters(advection_params)

        ps.Logger.getInstance().addInfo(f"Running simulation: {suffix}").print()
        process.apply()

        domain.saveSurfaceMesh(f"simpleEtching_{suffix}.vtp")

    run_simulation(ps.util.convertTemporalScheme("FORWARD_EULER"), False)
    run_simulation(ps.util.convertTemporalScheme("RUNGE_KUTTA_2ND_ORDER"), False)
    run_simulation(ps.util.convertTemporalScheme("RUNGE_KUTTA_2ND_ORDER"), True)
    run_simulation(ps.util.convertTemporalScheme("RUNGE_KUTTA_3RD_ORDER"), False)
    run_simulation(ps.util.convertTemporalScheme("RUNGE_KUTTA_3RD_ORDER"), True)


if __name__ == "__main__":
    main()
