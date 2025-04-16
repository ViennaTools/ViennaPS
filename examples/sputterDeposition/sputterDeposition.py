from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="sputterDeposition",
    description="Run sputter deposition based on varying deposition rates.",
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

vps.Logger.setLogLevel(vps.LogLevel.ERROR)
params = vps.ReadConfigFile(args.filename)
vps.setNumThreads(16)

geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
vps.MakeTrench(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["maskHeight"],
    taperingAngle=params["taperingAngle"],
    # baseHeight=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=vps.Material.Si,
).apply()

geometry.saveVolumeMesh("Trench")
geometry.duplicateTopLevelSet(vps.Material.SiO2)

direction = [0.0, 0.0, 0.0]
direction[vps.D - 1] = -1.0

# Define purely directional rate for depo removal
depoDir = vps.RateSet(
    direction=direction,
    calculateVisibility = True
)

Deposition = vps.DirectionalProcess(rateSets=[depoDir])

n = 0

def runDeposition(model, name, time=1.0):
    global n
    print("  - {} - ".format(name))
    vps.Process(geometry, model, time).apply()
    n += 1

numCycles = int(params["numCycles"])

# Initial geometry
geometry.saveSurfaceMesh("Deposition_{}".format(n))
n += 1

for i in range(numCycles):
    print("Cycle {}".format(i + 1))

    # Remove the polymer layer
    runDeposition(Deposition, "Deposition", params["depositionTime"])

    geometry.saveSurfaceMesh("Deposition_{}".format(n))
    n += 1

# save the final geometry
geometry.saveVolumeMesh("Final")
