from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(prog="DRAMWiggling", description="Run a DRAM etching process which results in AA wiggling.")
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps3d as vps

try:
    # ViennaLS Python bindings are needed for the extrusion tool
    import viennals3d as vls
except ModuleNotFoundError:
    print("ViennaLS Python module not found. Can not parse GDS file.")
    exit(1)

vps.Logger.setLogLevel(vps.LogLevel.ERROR)

gridDelta = 0.005

boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

mask = vps.GDSGeometry(gridDelta, boundaryConds)
reader = vps.GDSReader(mask, "wiggle_full.gds")
reader.apply()

# Prepare geometry
bounds = mask.getBounds()
geometry = vps.Domain()

# Substrate plane
origin = [0., 0., 0.]
normal = [0., 0., 1.]
substrate = vls.Domain(bounds, boundaryConds, gridDelta)
vls.MakeGeometry(substrate, vls.Plane(origin, normal)).apply()
geometry.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)
# Insert GDS layers
maskLS = mask.layerToLevelSet(0, 0.0, 0.18, True)
geometry.insertNextLevelSetAsMaterial(maskLS, vps.Material.Mask)

params = vps.ReadConfigFile(args.filename)

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
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

for i in range(1, 101):
    # run the process
    process.apply()
    geometry.saveSurfaceMesh(
        filename=f"etched_{i}.vtp", addMaterialIds=True
    )

# print final volume
geometry.saveVolumeMesh("Geometry")
