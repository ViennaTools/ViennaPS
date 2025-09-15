from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcessEmulate",
    description="Run a Bosch process emulation on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
else:
    print("Running 3D simulation.")
ps.setDimension(args.dim)

ps.Logger.setLogLevel(ps.LogLevel.ERROR)
params = ps.readConfigFile(args.filename)
ps.setNumThreads(16)

geometry = ps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
ps.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=0.0,
    maskHeight=params["maskHeight"],
).apply()


direction = [0.0, 0.0, 0.0]
direction[args.dim - 1] = -1.0

# Geometric advection model for deposition
depoModel = ps.SphereDistribution(
    radius=params["depositionThickness"], gridDelta=params["gridDelta"]
)

# Define purely directional rate for depo removal
etchDir = ps.RateSet(
    direction=direction,
    directionalVelocity=-(params["depositionThickness"] + params["gridDelta"] / 2.0),
    isotropicVelocity=0.0,
    maskMaterials=[ps.Material.Mask],
)
depoRemoval = ps.DirectionalProcess(rateSets=[etchDir])

# Define isotropic + direction rate for etching of substrate
etchIso = ps.RateSet(
    direction=direction,
    directionalVelocity=params["ionRate"],
    isotropicVelocity=params["neutralRate"],
    maskMaterials=[ps.Material.Mask, ps.Material.Polymer],
)
etchModel = ps.DirectionalProcess(rateSets=[etchIso])
etchTime = params["etchTime"]

n = 0


def runProcess(model, name, time=1.0):
    global n
    print("  - {} - ".format(name))
    ps.Process(geometry, model, time).apply()
    geometry.saveSurfaceMesh("boschProcessEmulate_{}".format(n))
    n += 1


def cleanup(threshold=1.0):
    expand = ps.IsotropicProcess(threshold)
    ps.Process(geometry, expand, 1).apply()
    shrink = ps.IsotropicProcess(-threshold)
    ps.Process(geometry, shrink, 1).apply()


numCycles = int(params["numCycles"])

# Initial geometry
geometry.saveSurfaceMesh("boschProcessEmulate_{}".format(n))
n += 1

runProcess(etchModel, "Etching", etchTime)

for i in range(numCycles):
    print("Cycle {}".format(i + 1))

    # Deposit a layer of polymer
    geometry.duplicateTopLevelSet(ps.Material.Polymer)
    runProcess(depoModel, "Deposition")

    # Remove the polymer layer
    runProcess(depoRemoval, "Punching through")

    # Etch the trench
    runProcess(etchModel, "Etching", etchTime)

    # Ash (remove) the polymer
    geometry.removeTopLevelSet()
    cleanup(params["gridDelta"])
    geometry.saveSurfaceMesh("boschProcessEmulate_{}".format(n))
    n += 1

# save the final geometry
geometry.saveVolumeMesh("boschProcessEmulate_final")
