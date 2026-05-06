from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(prog="holeEtching", description="Run a hole etching process.")
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
else:
    print("Running 3D simulation.")
ps.setDimension(args.dim)
ps.setNumThreads(16)

params = ps.readConfigFile(args.filename)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])


def run_simulation(fluxEngine, suffix):
    # geometry setup, all units in um
    geometry = ps.Domain(
        gridDelta=params["gridDelta"],
        xExtent=params["xExtent"],
        yExtent=params["yExtent"],
    )

    ps.MakeHole(
        domain=geometry,
        holeRadius=params["holeRadius"],
        holeDepth=0.0,
        maskHeight=params["maskHeight"],
        maskTaperAngle=params["taperAngle"],
        holeShape=ps.HoleShape.QUARTER,
    ).apply()

    # use pre-defined model SF6O2 etching model
    modelParams = ps.SF6O2Etching.defaultParameters()
    modelParams.ionFlux = params["ionFlux"]
    modelParams.etchantFlux = params["etchantFlux"]
    modelParams.passivationFlux = params["oxygenFlux"]
    modelParams.Ions.meanEnergy = params["meanEnergy"]
    modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
    modelParams.Ions.exponent = params["ionExponent"]
    modelParams.Passivation.A_ie = params["A_O"]
    modelParams.Substrate.A_ie = params["A_Si"]
    modelParams.etchStopDepth = params["etchStopDepth"]
    model = ps.SF6O2Etching(modelParams)

    covParams = ps.CoverageParameters()
    covParams.tolerance = 1e-4

    rayParams = ps.RayTracingParameters()
    rayParams.raysPerPoint = int(params["raysPerPoint"])

    advParams = ps.AdvectionParameters()
    advParams.spatialScheme = ps.util.convertSpatialScheme(params["spatialScheme"])
    advParams.temporalScheme = ps.util.convertTemporalScheme(params["temporalScheme"])

    # process setup
    process = ps.Process(geometry, model)
    process.setProcessDuration(params["processTime"])  # seconds
    process.setParameters(covParams)
    process.setParameters(rayParams)
    process.setParameters(advParams)

    process.setFluxEngineType(fluxEngine)

    # run the process
    process.apply()

    # print final surface
    output_file = "hole_"
    output_file += suffix
    geometry.saveSurfaceMesh(filename=output_file, addInterfaces=True)


run_simulation(ps.FluxEngineType.CPU_DISK, "CPU_disk")
run_simulation(ps.FluxEngineType.GPU_DISK, "GPU_disk")
run_simulation(ps.FluxEngineType.CPU_TRIANGLE, "CPU_triangle")
run_simulation(ps.FluxEngineType.GPU_TRIANGLE, "GPU_triangle")
