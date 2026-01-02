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
params = vps.readConfigFile(args.filename)

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

parameters = vps.FluorocarbonParameters()

matSi = vps.FluorocarbonMaterialParameters()
matSi.id = vps.Material.Si
matSi.density = 5.5

matSiO2 = vps.FluorocarbonMaterialParameters()
matSiO2.id = vps.Material.SiO2
matSiO2.density = 2.2

matSi3N4 = vps.FluorocarbonMaterialParameters()
matSi3N4.id = vps.Material.Si3N4
matSi3N4.density = 2.3

matPoly = vps.FluorocarbonMaterialParameters()
matPoly.id = vps.Material.Polymer
matPoly.density = 2.0
matPoly.beta_e = 0.6
matPoly.A_ie = 0.0361 * 2

matMask = vps.FluorocarbonMaterialParameters()
matMask.id = vps.Material.Mask
matMask.density = 500.0
matMask.beta_e = 0.1
matMask.beta_p = 0.01
matMask.Eth_sp = 20.0

parameters.addMaterial(matSi)
parameters.addMaterial(matSiO2)
parameters.addMaterial(matSi3N4)
parameters.addMaterial(matPoly)
parameters.addMaterial(matMask)

parameters.ionFlux = params["ionFlux"]
parameters.etchantFlux = params["etchantFlux"]
parameters.polyFlux = params["polyFlux"]
parameters.Ions.meanEnergy = params["meanIonEnergy"]
parameters.Ions.sigmaEnergy = params["sigmaIonEnergy"]
parameters.Ions.exponent = params["ionExponent"]

model = vps.d2.FluorocarbonEtching(parameters)

covParams = vps.CoverageParameters()
covParams.maxIterations = 10
covParams.tolerance = 1e-4

advParams = vps.AdvectionParameters()
advParams.spatialScheme = vps.SpatialScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER

process = vps.d2.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.setParameters(covParams)
process.setParameters(advParams)

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
