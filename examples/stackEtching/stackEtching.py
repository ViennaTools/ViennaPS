import viennaps2d as vps
from argparse import ArgumentParser

# parse config file name
parser = ArgumentParser(
    prog="stackEtching",
    description="Run a etching process on a stack of Si3N4/SiO2 layers.",
)
parser.add_argument("filename")
args = parser.parse_args()

# Set process verbosity
vps.Logger.setLogLevel(vps.LogLevel.INFO)

# Parse process parameters
params = vps.ReadConfigFile(args.filename)

vps.Length.setUnit(params["lengthUnit"])
vps.Time.setUnit(params["timeUnit"])

# Geometry setup
geometry = vps.Domain()
vps.MakeStack(
    geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    numLayers=int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius=0.0,
    trenchWidth=params["trenchWidth"],
    maskHeight=params["maskHeight"],
    periodicBoundary=False,
).apply()

geometry.duplicateTopLevelSet(vps.Material.Polymer)

model = vps.FluorocarbonEtching(
    ionFlux=params["ionFlux"],
    etchantFlux=params["etchantFlux"],
    polyFlux=params["polyFlux"],
    meanIonEnergy=params["meanIonEnergy"],
    sigmaIonEnergy=params["sigmaIonEnergy"],
    ionExponent=params["ionExponent"],
)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.setMaxCoverageInitIterations(10)
process.setTimeStepRatio(0.25)
process.setIntegrationScheme(vps.IntegrationScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER)

# print initial surface
geometry.saveVolumeMesh("initial")

process.apply()

# print final surface
geometry.saveVolumeMesh("final")

print("Extruding to 3D ...")
extruded = vps.Domain3D()
extrudeExtent = [-20.0, 20.0]
boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

vps.Extrude(geometry, extruded, extrudeExtent, 0, boundaryConds).apply()

extruded.saveHullMesh("final_extruded")
