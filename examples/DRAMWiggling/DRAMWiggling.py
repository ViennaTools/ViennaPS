import viennaps as ps
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="DRAMWiggling",
    description="Run a DRAM etching process which results in AA wiggling.",
)
parser.add_argument("filename")
args = parser.parse_args()

ps.setDimension(3)

gridDelta = 0.01 * (1.0 + 1e-12)
boundaryConds = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]

params = ps.readConfigFile(args.filename)

mask = ps.GDSGeometry(gridDelta, boundaryConds)
mask.setBoundaryPadding(0.1, 0.1)
reader = ps.GDSReader(mask, params["gdsFile"])
reader.apply()

# Prepare geometry
geometry = ps.Domain()

# Insert GDS layers
maskLS = mask.layerToLevelSet(0, 0.0, 0.18)
geometry.insertNextLevelSetAsMaterial(maskLS, ps.Material.Mask)

# Add plane
ps.MakePlane(geometry, 0.0, ps.Material.Si, True).apply()

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit(params["lengthUnit"])
ps.Time.setUnit(params["timeUnit"])

modelParams = ps.HBrO2Etching.defaultParameters()
modelParams.ionFlux = params["ionFlux"]
modelParams.etchantFlux = params["etchantFlux"]
modelParams.passivationFlux = params["oxygenFlux"]
modelParams.Ions.meanEnergy = params["meanEnergy"]
modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
modelParams.Ions.exponent = params["ionExponent"]
modelParams.Ions.n_l = 200
model = ps.HBrO2Etching(modelParams)

coverageParameters = ps.CoverageParameters()
coverageParameters.tolerance = 1e-5

rayTracingParams = ps.RayTracingParameters()
rayTracingParams.raysPerPoint = int(params["raysPerPoint"])

advectionParams = ps.AdvectionParameters()
advectionParams.spatialScheme = ps.util.convertSpatialScheme(
    params["spatialScheme"]
)

fluxEngineStr = params["fluxEngine"]
fluxEngine = ps.util.convertFluxEngineType(fluxEngineStr)

# process setup
process = ps.Process(geometry, model)
process.setProcessDuration(params["processTime"])  # seconds
process.setParameters(coverageParameters)
process.setParameters(rayTracingParams)
process.setParameters(advectionParams)
process.setFluxEngineType(fluxEngine)

# print initial surface
geometry.saveSurfaceMesh(filename=f"DRAM_Initial_{fluxEngineStr}.vtp")

numSteps = int(params["numSteps"])
for i in range(numSteps):
    # run the process
    process.apply()
    geometry.saveSurfaceMesh(filename=f"DRAM_Etched_{fluxEngineStr}_{i + 1}.vtp")

# print final volume
geometry.saveHullMesh(f"DRAM_Final_{fluxEngineStr}")
