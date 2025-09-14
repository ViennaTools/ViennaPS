import viennaps as vps
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
geometry = vps.d2.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
vps.d2.MakeStack(
    domain=geometry,
    numLayers=int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius=0.0,
    trenchWidth=params["trenchWidth"],
    maskHeight=params["maskHeight"],
).apply()

geometry.duplicateTopLevelSet(vps.Material.Polymer)

model = vps.d2.FluorocarbonEtching(
    ionFlux=params["ionFlux"],
    etchantFlux=params["etchantFlux"],
    polyFlux=params["polyFlux"],
    meanIonEnergy=params["meanIonEnergy"],
    sigmaIonEnergy=params["sigmaIonEnergy"],
    ionExponent=params["ionExponent"],
)

covParams = vps.CoverageParameters()
covParams.maxIterations = 10

advParams = vps.AdvectionParameters()
advParams.integrationScheme = vps.IntegrationScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER

process = vps.d2.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.setCoverageParameters(covParams)
process.setAdvectionParameters(advParams)

# print initial surface
geometry.saveVolumeMesh("initial")

process.apply()

# print final surface
geometry.saveVolumeMesh("final")

print("Extruding to 3D ...")
extruded = vps.d3.Domain()
extrudeExtent = [-20.0, 20.0]
boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

vps.Extrude(geometry, extruded, extrudeExtent, 1, boundaryConds).apply()

extruded.saveHullMesh("final_extruded")
