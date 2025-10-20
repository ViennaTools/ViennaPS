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

params = ps.readConfigFile(args.filename)

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

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
    holeShape=ps.HoleShape.HALF,
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
advParams.integrationScheme = ps.util.convertIntegrationScheme(
    params["integrationScheme"]
)

# process setup
process = ps.Process(geometry, model)
process.setProcessDuration(params["processTime"])  # seconds
process.setParameters(covParams)
process.setParameters(rayParams)
process.setParameters(advParams)
if ps.gpuAvailable() and args.dim == 3:
    process.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp")

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp")
