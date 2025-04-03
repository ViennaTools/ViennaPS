from argparse import ArgumentParser
import viennaps3d as vps

# parse config file name and simulation dimension
parser = ArgumentParser(prog="hole", description="Run a hole etching process.")
parser.add_argument("filename")
args = parser.parse_args()

params = vps.ReadConfigFile(args.filename)

# print intermediate output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.INFO)
vps.setNumThreads(16)

vps.Length.setUnit(params["lengthUnit"])
vps.Time.setUnit(params["timeUnit"])

context = vps.gpu.Context()
context.create()

# geometry setup, all units in um
geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
vps.MakeHole(
    domain=geometry,
    holeRadius=params["holeRadius"],
    holeDepth=0.0,
    maskHeight=params["maskHeight"],
    maskTaperAngle=params["taperAngle"],
    holeShape=vps.HoleShape.Half,
).apply()

# use pre-defined model SF6O2 etching model
modelParams = vps.SF6O2Parameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.oxygenFlux = params["oxygenFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Ions.exponent = params["ionExponent"]
modelParams.Passivation.A_ie = params["A_O"]
modelParams.Si.A_ie = params["A_Si"]
modelParams.etchStopDepth = params["etchStopDepth"]
model = vps.gpu.SF6O2Etching(modelParams)

rayTracingParams = vps.RayTracingParameters()
rayTracingParams.smoothingNeighbors = 2
rayTracingParams.raysPerPoint = int(params["raysPerPoint"])

# process setup
process = vps.gpu.Process(context)
process.setDomain(geometry)
process.setProcessModel(model)
# process.setMaxCoverageInitIterations(20)
process.setNumberOfRaysPerPoint(int(params["raysPerPoint"]))
process.setProcessDuration(params["processTime"])  # seconds
process.setIntegrationScheme(
    vps.util.convertIntegrationScheme(params["integrationScheme"])
)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
