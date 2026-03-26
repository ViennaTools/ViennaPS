from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="trenchDeposition",
    description="Run a deposition process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

if args.dim == 2:
    print("Running 2D simulation.")
else:
    print("Running 3D simulation.")
ps.setDimension(args.dim)

params = ps.readConfigFile(args.filename)


def run_simulation(fluxEngine, suffix):
    geometry = ps.Domain(
        gridDelta=params["gridDelta"] / 3.5,
        xExtent=params["xExtent"],
        yExtent=params["yExtent"],
    )
    ps.MakeTrench(
        domain=geometry,
        trenchWidth=params["trenchWidth"],
        trenchDepth=params["trenchHeight"] * 3,
        trenchTaperAngle=params["taperAngle"],
    ).apply()

    geometry.duplicateTopLevelSet(ps.Material.SiO2)

    model = ps.SingleParticleProcess(
        stickingProbability=0.001,
        sourceExponent=params["sourcePower"],
    )

    process = ps.Process(geometry, model)
    process.setProcessDuration(params["processTime"])
    process.setFluxEngineType(fluxEngine)

    process.apply()

    filename = f"trench_{suffix}.vtp"
    geometry.saveSurfaceMesh(filename=filename, addInterfaces=True)


run_simulation(ps.FluxEngineType.CPU_DISK, "CPU_disk")
run_simulation(ps.FluxEngineType.GPU_DISK, "GPU_disk")
run_simulation(ps.FluxEngineType.CPU_TRIANGLE, "CPU_triangle")
run_simulation(ps.FluxEngineType.GPU_TRIANGLE, "GPU_triangle")
