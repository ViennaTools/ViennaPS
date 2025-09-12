from argparse import ArgumentParser
import viennaps.d3 as psd
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(prog="hole", description="Run a hole etching process.")
parser.add_argument("filename")
args = parser.parse_args()

params = ps.ReadConfigFile(args.filename)

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)
ps.setNumThreads(16)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

ps.gpu.Context.createContext()

# geometry setup, all units in um
geometry = psd.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
psd.MakeHole(
    domain=geometry,
    holeRadius=params["holeRadius"],
    holeDepth=0.0,
    maskHeight=params["maskHeight"],
    maskTaperAngle=params["taperAngle"],
    holeShape=ps.HoleShape.HALF,
).apply()

# use pre-defined model SF6O2 etching model
modelParams = psd.SF6O2Etching.defaultParameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.passivationFlux = params["oxygenFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Ions.exponent = params["ionExponent"]
modelParams.Passivation.A_ie = params["A_O"]
modelParams.Substrate.A_ie = params["A_Si"]
modelParams.etchStopDepth = params["etchStopDepth"]
model = psd.SF6O2Etching(modelParams)

rayTracingParams = ps.RayTracingParameters()
rayTracingParams.smoothingNeighbors = 2
rayTracingParams.raysPerPoint = int(params["raysPerPoint"])

advectionParams = ps.AdvectionParameters()
advectionParams.integrationScheme = ps.util.convertIntegrationScheme(
    params["integrationScheme"]
)

# process setup
process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])  # seconds
process.setRayTracingParameters(rayTracingParams)
process.setAdvectionParameters(advectionParams)
process.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
