from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcess",
    description="Run a Bosch process simulation on a trench geometry.",
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
vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vps.setNumThreads(16)

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

# Isoptropic deposition model
depoModel = vps.SingleParticleProcess(
    rate=params["depositionThickness"],
    stickingProbability=params["depositionStickingProbability"],
)

# Deposition removal model
depoRemoval = vps.SingleParticleProcess(
    rate=-params["depositionThickness"],
    stickingProbability=1.0,
    sourceExponent=params["ionSourceExponent"],
    maskMaterial=vps.Material.Mask,
)

# Etch model
etchModel = vps.MultiParticleProcess()
etchModel.addNeutralParticle(params["neutralStickingProbability"])
etchModel.addIonParticle(sourcePower=params["ionSourceExponent"], thetaRMin=60.0)


# Custom rate function for the etch model
def rateFunction(fluxes, material):
    if material == vps.Material.Mask:
        return 0.0
    rate = fluxes[1] * params["ionRate"]
    if material == vps.Material.Si:
        rate += fluxes[0] * params["neutralRate"]
    return rate


etchModel.setRateFunction(rateFunction)
etchTime = params["etchTime"]

n = 0


def runProcess(model, name, time=1.0):
    global n
    print("  - {} - ".format(name))
    vps.Process(geometry, model, time).apply()
    geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
    n += 1


def cleanup(threshold=1.0):
    expand = vps.IsotropicProcess(threshold)
    vps.Process(geometry, expand, 1).apply()
    shrink = vps.IsotropicProcess(-threshold)
    vps.Process(geometry, shrink, 1).apply()


numCycles = int(params["numCycles"])

# Initial geometry
geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
n += 1

runProcess(etchModel, "Etching", etchTime)

for i in range(numCycles):
    print("Cycle {}".format(i + 1))

    # Deposit a layer of polymer
    geometry.duplicateTopLevelSet(vps.Material.Polymer)
    runProcess(depoModel, "Deposition")

    # Remove the polymer layer
    runProcess(depoRemoval, "Punching through")

    # Etch the trench
    runProcess(etchModel, "Etching", etchTime)

    # Ash (remove) the polymer
    geometry.removeTopLevelSet()
    cleanup(params["gridDelta"])
    geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
    n += 1

# save the final geometry
geometry.saveVolumeMesh("boschProcessSimulate_final")
