import viennaps2d as vps
from SiGeStackGeometry import CreateGeometry
import numpy as np

vps.setNumThreads(16)

# create initial geometry
paramDict = {
    "numPillars":5,
    "numLayers": 12,
    "layerHeight": 20.0,
    "maskWidth": 100.0,
    "maskHeight": 55.0,
    "trenchWidthTop": 100.0,
    "trenchWidthBottom": 100.0,
    "overEtch": 100.0,
    "lateralSpacing": 300.0,
    "periodicBoundary": False,
    "gridDelta": 2.0,
}
geometry = CreateGeometry(paramDict)

config_file = "config_CF4O2.txt"
params = vps.ReadConfigFile(config_file)

vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

vps.Length.setUnit(params["lengthUnit"])
vps.Time.setUnit(params["timeUnit"])

# use pre-defined model CF4O2 etching model
modelParams = vps.CF4O2Parameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.oxygenFlux = params["oxygenFlux"]
modelParams.polymerFlux = params["polymerFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Passivation.A_O_ie = params["A_O"]
modelParams.Passivation.A_C_ie = params["A_C"]
modelParams.etchStopDepth = params["etchStopDepth"]

# Use Material enum
modelParams.gamma_F = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.1,
    vps.Material.SiGe: 0.1
}
modelParams.gamma_F_oxidized = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.1,
    vps.Material.SiGe: 0.1
}
modelParams.gamma_O = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.7,
    vps.Material.SiGe: 0.7
}
modelParams.gamma_O_passivated = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.7,
    vps.Material.SiGe: 0.7
}
modelParams.gamma_C = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.7,
    vps.Material.SiGe: 0.7
}
modelParams.gamma_C_oxidized = {
    vps.Material.Mask: 0.0,
    vps.Material.Si: 0.7,
    vps.Material.SiGe: 0.7
}

model = vps.CF4O2Etching(modelParams)
parameters = model.getParameters()

# process setup
process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setMaxCoverageInitIterations(10)
process.setNumberOfRaysPerPoint(int(params["raysPerPoint"]))
process.setProcessDuration(params["processTime"])  # seconds
process.setIntegrationScheme(
    vps.util.convertIntegrationScheme(params["integrationScheme"])
)
process.setTimeStepRatio(0.2)

# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)
geometry.saveVolumeMesh("initial")

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
geometry.saveVolumeMesh("final")
