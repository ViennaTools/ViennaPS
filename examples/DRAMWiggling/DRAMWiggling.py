import viennaps.d3 as psd
import viennaps as ps
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="DRAMWiggling",
    description="Run a DRAM etching process which results in AA wiggling.",
)
parser.add_argument("filename")
args = parser.parse_args()

gridDelta = 0.01 * (1.0 + 1e-12)
boundaryConds = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]

params = ps.readConfigFile(args.filename)

mask = psd.GDSGeometry(gridDelta, boundaryConds)
mask.setBoundaryPadding(0.1, 0.1)
reader = psd.GDSReader(mask, params["gdsFile"])
reader.apply()

# Prepare geometry
geometry = psd.Domain()

# Insert GDS layers
maskLS = mask.layerToLevelSet(0, 0.0, 0.18)
geometry.insertNextLevelSetAsMaterial(maskLS, ps.Material.Mask)

# Add plane
psd.MakePlane(geometry, 0.0, ps.Material.Si, True).apply()

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

modelParams = psd.HBrO2Etching.defaultParameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.passivationFlux = params["oxygenFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Ions.exponent = params["ionExponent"]
modelParams.Ions.n_l = 200
model = psd.HBrO2Etching(modelParams)

coverageParameters = ps.CoverageParameters()
coverageParameters.maxIterations = 10

rayTracingParams = ps.RayTracingParameters()
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
process.setCoverageParameters(coverageParameters)
process.setRayTracingParameters(rayTracingParams)
process.setAdvectionParameters(advectionParams)

# print initial surface
geometry.saveSurfaceMesh(filename="DRAM_Initial.vtp", addMaterialIds=True)

numSteps = int(params["numSteps"])
for i in range(numSteps):
    # run the process
    process.apply()
    geometry.saveSurfaceMesh(filename=f"DRAM_Etched_{i + 1}.vtp", addMaterialIds=True)

# print final volume
geometry.saveHullMesh("DRAM_Final")
