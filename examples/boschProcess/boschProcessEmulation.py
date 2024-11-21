from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcess",
    description="Run a Bosch process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

params = vps.ReadConfigFile(args.filename)
vps.Logger.setLogLevel(vps.LogLevel.INFO)

geometry = vps.Domain()
vps.MakeTrench(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["maskHeight"],
    taperingAngle=0.0,
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=True,
    material=vps.Material.Si,
).apply()


direction = [0.0, 0.0, 0.0]
direction[vps.D - 1] = -1.0

depoModel = vps.IsotropicProcess(params["depositionThickness"])
depoRemoval = vps.DirectionalEtching(
    direction,
    params["depositionThickness"] + params["gridDelta"],
    0.0,
    True,
    vps.Material.Mask,
)
etchModel = vps.DirectionalEtching(
    direction,
    params["ionRate"],
    params["neutralRate"],
    False,
    [vps.Material.Mask, vps.Material.Polymer],
)


geometry.saveSurfaceMesh("initial.vtp")

proc = vps.Process(geometry, etchModel, params["etchTime"])
proc.disableRandomSeeds()
proc.apply()

numCycles = int(params["numCycles"])
for i in range(numCycles):
    geometry.duplicateTopLevelSet(vps.Material.Polymer)
    vps.Process(geometry, depoModel, 1).apply()
    vps.Process(geometry, depoRemoval, 1).apply()
    vps.Process(geometry, etchModel, params["etchTime"]).apply()
    geometry.removeTopLevelSet()

geometry.saveSurfaceMesh("final.vtp")

if args.dim == 2:
    geometry.saveVolumeMesh("final")
