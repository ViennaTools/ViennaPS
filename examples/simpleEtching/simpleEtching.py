import viennaps
import viennaps.d2 as vps2
import viennaps.d3 as vps3

def main():
    # Dimension of the domain: 2 for Trench, 3 for Hole
    D = 2
    
    viennaps.Logger.setLogLevel(viennaps.LogLevel.INFO)

    # Geometry parameters
    grid_delta = 0.05
    x_extent = 3.0
    y_extent = 3.0
    feature_width = 1.0 # Diameter for hole, Width for trench
    mask_height = 1.0
    taper_angle = 0.0

    # Single Particle Process Model
    # Parameters: rate, stickingProbability, yield, maskMaterial
    rate = -2.0
    sticking_probability = 0.2
    source_power = 1.0
    process_time = 1.0

    def run_simulation(temporal_scheme, calc_intermediate):
        suffix = str(temporal_scheme).split(".")[-1]
        if calc_intermediate:
            suffix += "_recalc"
        if D == 3:
            # Create a Hole in 3D
            domain = vps3.Domain(grid_delta, x_extent, y_extent)
            vps3.MakeHole(domain, feature_width / 2.0, 0.0, 0.0,
                          mask_height, taper_angle,
                          viennaps.HoleShape.QUARTER).apply()
            
            model = vps3.SingleParticleProcess(rate, sticking_probability, source_power, viennaps.Material.Mask)
            process = vps3.Process(domain, model, process_time)
        else:
            # Create a Trench in 2D
            domain = vps2.Domain(grid_delta, x_extent, y_extent)
            vps2.MakeTrench(domain, feature_width, 0.0, 0.0, mask_height,
                            taper_angle, False).apply()
            
            model = vps2.SingleParticleProcess(rate, sticking_probability, source_power, viennaps.Material.Mask)
            process = vps2.Process(domain, model, process_time)

        advection_params = viennaps.AdvectionParameters()
        advection_params.spatialScheme = viennaps.util.convertSpatialScheme("WENO_5TH_ORDER")
        advection_params.temporalScheme = temporal_scheme
        advection_params.calculateIntermediateVelocities = calc_intermediate
        process.setParameters(advection_params)

        viennaps.Logger.getInstance().addInfo(f"Running simulation: {suffix}").print()
        process.apply()

        domain.saveSurfaceMesh(f"simpleEtching_{suffix}.vtp")

    run_simulation(viennaps.util.convertTemporalScheme("FORWARD_EULER"), False)
    run_simulation(viennaps.util.convertTemporalScheme("RUNGE_KUTTA_2ND_ORDER"), False)
    run_simulation(viennaps.util.convertTemporalScheme("RUNGE_KUTTA_2ND_ORDER"), True)
    run_simulation(viennaps.util.convertTemporalScheme("RUNGE_KUTTA_3RD_ORDER"), False)
    run_simulation(viennaps.util.convertTemporalScheme("RUNGE_KUTTA_3RD_ORDER"), True)

if __name__ == "__main__":
    main()
    