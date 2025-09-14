from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(prog="holeEtching", description="Run a hole etching process.")
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps as ps

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps.d2 as psd
else:
    print("Running 3D simulation.")
    import viennaps.d3 as psd

params = ps.ReadConfigFile(args.filename)

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

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

covParams = ps.CoverageParameters()
covParams.maxIterations = 20

rayParams = ps.RayTracingParameters()
rayParams.raysPerPoint = int(params["raysPerPoint"])

advParams = ps.AdvectionParameters()
advParams.integrationScheme = ps.util.convertIntegrationScheme(
    params["integrationScheme"]
)

# process setup
process = psd.Process(geometry, model)
process.setProcessDuration(params["processTime"])  # seconds
process.setCoverageParameters(covParams)
process.setRayTracingParameters(rayParams)
process.setAdvectionParameters(advParams)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
