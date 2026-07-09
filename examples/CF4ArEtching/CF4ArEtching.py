# CF4/Ar silicon etching benchmark example.
#
# Uses the dedicated CF4/Ar-on-Si model (ps.CF4ArEtching), which is a thin
# configuration of the generic ViennaPS PlasmaEtching framework. Because it
# reuses that framework it runs on both the CPU and the GPU ray tracer.
#
# Species / surface-state mapping:
#   Ion         -> Ar+           (ionFlux)
#   Etchant     -> F   radical   (etchantFlux),     coverage theta_F
#   Passivation -> lumped CFx    (passivationFlux), coverage theta_CF
#
# Staged chemistry from the CF4/Ar benchmark plan:
#   Step 1 : F + Ar+          -> passivationFlux = 0
#   Step 2 : add lumped CFx   -> passivationFlux > 0 (sidewall passivation)
#
# Geometry:
#   2D (default) -> masked Si trench
#   3D           -> masked Si cylindrical hole
# Select with:  python CF4ArEtching.py -D 3

from argparse import ArgumentParser

import viennaps as ps

parser = ArgumentParser(
    prog="CF4ArEtching", description="CF4/Ar Si etching (2D trench or 3D hole)."
)
parser.add_argument("-D", "--dim", dest="dim", type=int, default=2, choices=(2, 3))
args = parser.parse_args()

ps.setDimension(args.dim)
ps.setNumThreads(16)
ps.Logger.setLogLevel(ps.LogLevel.INFO)

# feature-scale units
ps.Length.setUnit("nm")
ps.Time.setUnit("s")

# --- geometry parameters (all lengths in nm) -------------------------------
grid_delta = 2.0
x_extent = 200.0
y_extent = 200.0
feature_width = 80.0  # trench width (2D) / hole diameter (3D)
feature_depth = 0.0  # initial etched depth of the feature
mask_height = 60.0

process_time = 10.0  # seconds (blanket Si rate ~14 nm/s at these fluxes)

print(f"Running {args.dim}D simulation "
      f"({'trench' if args.dim == 2 else 'cylindrical hole'}).")

# --- ray-tracing engine ----------------------------------------------------
# Prefer the GPU triangle-mesh flux engine; fall back to the CPU engine if the
# build has no GPU support.
if ps.gpuAvailable():
    flux_engine = ps.FluxEngineType.GPU_TRIANGLE
    print("Using GPU ray tracing (GPU_TRIANGLE).")
else:
    flux_engine = ps.FluxEngineType.CPU_DISK
    print("GPU not available - using CPU ray tracing (CPU_DISK).")


def make_geometry():
    domain = ps.Domain(gridDelta=grid_delta, xExtent=x_extent, yExtent=y_extent)
    if args.dim == 2:
        ps.MakeTrench(
            domain=domain,
            trenchWidth=feature_width,
            trenchDepth=feature_depth,
            maskHeight=mask_height,
            material=ps.Material.Si,
            maskMaterial=ps.Material.Mask,
        ).apply()
    else:
        # 3D cylindrical hole (quarter geometry exploits the 4-fold symmetry).
        ps.MakeHole(
            domain=domain,
            holeRadius=feature_width / 2.0,
            holeDepth=feature_depth,
            maskHeight=mask_height,
            holeShape=ps.HoleShape.QUARTER,
            material=ps.Material.Si,
            maskMaterial=ps.Material.Mask,
        ).apply()
    return domain


def base_parameters():
    # PlasmaEtchingParameters tuned for CF4/Ar (Step 1 baseline).
    params = ps.CF4ArEtching.defaultParameters()
    params.ionFlux = 12.0  # Ar+ ion flux
    params.etchantFlux = 1.8e2  # F radical flux (reduced for a moderate rate)
    params.passivationFlux = 0.0  # no CFx residue (Step 1)
    params.Ions.meanEnergy = 100.0  # eV
    params.Ions.sigmaEnergy = 10.0  # eV
    params.Ions.exponent = 500.0
    return params


def run(params, name):
    domain = make_geometry()
    model = ps.CF4ArEtching(params)

    process = ps.Process(domain, model, process_time)
    process.setFluxEngineType(flux_engine)

    cov_params = ps.CoverageParameters()
    cov_params.tolerance = 1e-4
    process.setParameters(cov_params)

    ray_params = ps.RayTracingParameters()
    ray_params.raysPerPoint = 1000
    process.setParameters(ray_params)

    # Surface-diffusion stencil parameters (defaults are usually fine); the
    # diffusion itself is enabled per-coverage via the *DiffusionCoefficient
    # fields on the parameters.
    process.setParameters(ps.SurfaceDiffusionParameters())

    ps.Logger.getInstance().addInfo(f"Running: {name}").print()
    process.apply()
    domain.saveSurfaceMesh(filename=f"CF4Ar_{args.dim}D_{name}.vtp", addInterfaces=True)


def main():
    make_geometry().saveSurfaceMesh(filename=f"CF4Ar_{args.dim}D_initial.vtp")

    # Step 1: F + Ar+ baseline
    run(base_parameters(), "step1_F_Ar")

    # Step 2: add lumped CFx residue (sidewall passivation)
    params = base_parameters()
    params.passivationFlux = 1.0e2  # turn on the fluorocarbon channel
    params.Passivation.A_ie = 3.0  # Ar+ residue-removal yield
    params.Passivation.Eth_ie = 10.0  # residue removal threshold (eV)
    run(params, "step2_CFx")

    # Step 1 + surface diffusion of the fluorine coverage theta_F.
    # Setting a positive coefficient enables operator-split diffusion of the
    # "eCoverage" (theta_F) field after each local reaction update.
    params = base_parameters()
    params.etchantDiffusionCoefficient = 5.0e3  # D_F in nm^2/s
    run(params, "step1_F_diffusion")


if __name__ == "__main__":
    main()
