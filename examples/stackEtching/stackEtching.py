import viennaps2d as vps
from argparse import ArgumentParser

# parse config file name
parser = ArgumentParser(
    prog="stackEtching",
    description="Run a etching process on a stack of Si3N4/SiO2 layers.",
)
parser.add_argument("filename")
args = parser.parse_args()

extrude = True
try:
    # ViennaLS Python bindings are needed for the extrusion tool
    import viennals3d as vls
except ModuleNotFoundError:
    print("ViennaLS Python module not found. Can not extrude.")
    extrude = False

# Set process verbosity
vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)

# Parse process parameters
params = vps.ReadConfigFile(args.filename)

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

# print initial surface
geometry.saveVolume("initial")

process.apply()

# print final surface
geometry.saveVolume("final")

if extrude:
    print("Extruding to 3D ...")
    extruded = vps.Domain3D()
    extrudeExtent = [-20.0, 20.0]
    boundaryConds = [
        vls.lsBoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.lsBoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.lsBoundaryConditionEnum.INFINITE_BOUNDARY,
    ]

    vps.Extrude(geometry, extruded, extrudeExtent, 0, boundaryConds).apply()

    extruded.saveSurface("extruded_surface.vtp", True)
    extruded.saveVolume("extruded_volume")
