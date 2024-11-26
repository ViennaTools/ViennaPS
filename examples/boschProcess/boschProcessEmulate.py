from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcessEmulate",
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

# Define directional rate
etchDir = vps.RateSet(
    direction=direction,
    directionalVelocity=-(params["depositionThickness"] + params["gridDelta"] / 2.0),
    isotropicVelocity=0.0,
    maskMaterials=[vps.Material.Mask],
)

# Define isotropic rate
etchIso = vps.RateSet(
    direction=direction,
    directionalVelocity=params["ionRate"],
    isotropicVelocity=params["neutralRate"],
    maskMaterials=[vps.Material.Mask, vps.Material.Polymer],
)

depoRemoval = vps.DirectionalEtching(rateSets=[etchDir])

etchModel = vps.DirectionalEtching(rateSets=[etchIso])

numCycles = int(params["numCycles"])
n = 0

geometry.saveSurfaceMesh("boschProcess_{}".format(n))
n += 1

vps.Process(geometry, etchModel, params["etchTime"]).apply()
geometry.saveSurfaceMesh("boschProcess_{}".format(n))
n += 1

for i in range(numCycles):
    # Deposit a layer of polymer
    geometry.duplicateTopLevelSet(vps.Material.Polymer)
    vps.Process(geometry, depoModel, 1).apply()
    geometry.saveSurfaceMesh("boschProcess_{}".format(n))
    n += 1

    # Remove the polymer layer
    vps.Process(geometry, depoRemoval, 1).apply()
    geometry.saveSurfaceMesh("boschProcess_{}".format(n))
    n += 1

    # Etch the trench
    vps.Process(geometry, etchModel, params["etchTime"]).apply()
    geometry.saveSurfaceMesh("boschProcess_{}".format(n))
    n += 1

    # Ash the polymer
    geometry.removeTopLevelSet()
    geometry.saveSurfaceMesh("boschProcess_{}".format(n))
    n += 1

geometry.saveVolumeMesh("final")
