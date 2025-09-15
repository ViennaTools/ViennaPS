from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcess",
    description="Run a Bosch process simulation on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps as ps

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    psd = ps.d2
else:
    print("Running 3D simulation.")
    psd = ps.d3

ps.Logger.setLogLevel(ps.LogLevel.ERROR)
params = ps.ReadConfigFile(args.filename)
ps.setNumThreads(16)

geometry = psd.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
psd.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=0.0,
    maskHeight=params["maskHeight"],
).apply()

# Isotropic deposition model
depoModel = psd.SingleParticleProcess(
    rate=params["depositionThickness"],
    stickingProbability=params["depositionStickingProbability"],
)

# Deposition removal model
depoRemoval = psd.SingleParticleProcess(
    rate=-params["depositionThickness"],
    stickingProbability=1.0,
    sourceExponent=params["ionSourceExponent"],
    maskMaterial=ps.Material.Mask,
)

# Etch model
etchModel = psd.MultiParticleProcess()
etchModel.addNeutralParticle(params["neutralStickingProbability"])
etchModel.addIonParticle(sourcePower=params["ionSourceExponent"], thetaRMin=60.0)


# Custom rate function for the etch model
def rateFunction(fluxes, material):
    if material == ps.Material.Mask:
        return 0.0
    rate = fluxes[1] * params["ionRate"]
    if material == ps.Material.Si:
        rate += fluxes[0] * params["neutralRate"]
    return rate


etchModel.setRateFunction(rateFunction)
etchTime = params["etchTime"]

n = 0


def runProcess(model, name, time=1.0):
    global n
    print("  - {} - ".format(name))
    psd.Process(geometry, model, time).apply()
    geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
    n += 1


def cleanup(threshold=1.0):
    expand = psd.IsotropicProcess(threshold)
    psd.Process(geometry, expand, 1).apply()
    shrink = psd.IsotropicProcess(-threshold)
    psd.Process(geometry, shrink, 1).apply()


numCycles = int(params["numCycles"])

# Initial geometry
geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
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
    geometry.saveSurfaceMesh("boschProcessSimulate_{}".format(n))
    n += 1

# save the final geometry
geometry.saveVolumeMesh("boschProcessSimulate_final")
