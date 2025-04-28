import viennaps3d as vps
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="DRAMWiggling",
    description="Run a DRAM etching process which results in AA wiggling.",
)
parser.add_argument("filename")
args = parser.parse_args()

gridDelta = 0.005 * (1.0 + 1e-12)
boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

params = vps.ReadConfigFile(args.filename)

mask = vps.GDSGeometry(gridDelta, boundaryConds)
mask.setBoundaryPadding(0.1, 0.1)
reader = vps.GDSReader(mask, params["gdsFile"])
reader.apply()

# Prepare geometry
geometry = vps.Domain()

# Insert GDS layers
maskLS = mask.layerToLevelSet(0, 0.0, 0.18, True)
geometry.insertNextLevelSetAsMaterial(maskLS, vps.Material.Mask)

# Add plane
vps.MakePlane(geometry, 0.0, vps.Material.Si, True).apply()

# print intermediate output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.INFO)

vps.Length.setUnit(params["lengthUnit"])
vps.Time.setUnit(params["timeUnit"])

modelParams = vps.HBrO2Parameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.oxygenFlux = params["oxygenFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Ions.exponent = params["ionExponent"]
modelParams.Ions.n_l = 200
model = vps.HBrO2Etching(modelParams)

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

# print initial surface
geometry.saveSurfaceMesh(filename="DRAM_Initial.vtp", addMaterialIds=True)

numSteps = int(params["numSteps"])
for i in range(numSteps):
    # run the process
    process.apply()
    geometry.saveSurfaceMesh(filename=f"DRAM_Etched_{i + 1}.vtp", addMaterialIds=True)

# print final volume
geometry.saveHullMesh("DRAM_Final")
