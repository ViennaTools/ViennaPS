import viennaps.d2 as psd
import viennaps as ps
from SiGeStackGeometry import CreateGeometry

ps.setNumThreads(16)

# create initial geometry
paramDict = {
    "numPillars": 3,
    "numLayers": 12,
    "layerHeight": 20.0,
    "maskWidth": 100.0,
    "maskHeight": 55.0,
    "trenchWidthTop": 100.0,
    "trenchWidthBottom": 100.0,
    "overEtch": 100.0,
    "lateralSpacing": 300.0,
    "periodicBoundary": False,
    "gridDelta": 2.5,
}
geometry = CreateGeometry(paramDict)

config_file = "config_CF4O2.txt"
params = ps.readConfigFile(config_file)

ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

# use pre-defined model CF4O2 etching model
modelParams = ps.CF4O2Parameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.oxygenFlux = params["oxygenFlux"]
modelParams.polymerFlux = params["polymerFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Passivation.A_O_ie = params["A_O"]
modelParams.Passivation.A_C_ie = params["A_C"]

# Use Material enum
modelParams.gamma_F = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.1,
    ps.Material.SiGe: 0.1,
}
modelParams.gamma_F_oxidized = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.1,
    ps.Material.SiGe: 0.1,
}
modelParams.gamma_O = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.7,
    ps.Material.SiGe: 0.7,
}
modelParams.gamma_O_passivated = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.7,
    ps.Material.SiGe: 0.7,
}
modelParams.gamma_C = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.7,
    ps.Material.SiGe: 0.7,
}
modelParams.gamma_C_oxidized = {
    ps.Material.Mask: 0.0,
    ps.Material.Si: 0.7,
    ps.Material.SiGe: 0.7,
}

model = psd.CF4O2Etching(modelParams)
parameters = model.getParameters()

covParams = ps.CoverageParameters()
covParams.maxIterations = 20
covParams.coverageDeltaThreshold = 1e-4

rayParams = ps.RayTracingParameters()
rayParams.raysPerPoint = int(params["raysPerPoint"])

advParams = ps.AdvectionParameters()
advParams.timeStepRatio = 0.2

# process setup
process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])  # seconds
process.setCoverageParameters(covParams)
process.setRayTracingParameters(rayParams)
process.setAdvectionParameters(advParams)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)
geometry.saveVolumeMesh("initial")

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
geometry.saveVolumeMesh("final")
