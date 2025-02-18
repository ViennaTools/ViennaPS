from argparse import ArgumentParser

# parse config file name
parser = ArgumentParser(
    prog="trenchDepositionGeometric",
    description="Run a geometric deposition process on a trench geometry.",
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

geometry = vps.Domain()
vps.MakeTrench(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    taperingAngle=params["taperAngle"],
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=vps.Material.Si,
).apply()

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(vps.Material.SiO2)

model = vps.SphereDistribution(
    radius=params["layerThickness"], gridDelta=params["gridDelta"]
)

geometry.saveHullMesh("initial")

vps.Process(geometry, model, 0.0).apply()

geometry.saveHullMesh("final")
