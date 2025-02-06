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

# Isoptropic deposition model (emulation of the deposition process)
depoModel = vps.IsotropicProcess(params["depositionThickness"])

depoRemoval = vps.SingleParticleProcess(
    -(params["depositionThickness"] + params["gridDelta"] / 2.0),  # rate
    1.0,  # sticking probability
    params["ionSourceExponent"],  # source exponent
    vps.Material.Mask,  # mask material
)

etchModel = vps.MultiParticleProcess()
etchModel.addNeutralParticle(params["neutralStickingProbability"])
etchModel.addIonParticle(params["ionSourceExponent"])


# Custom rate function for the etch model
def rateFunction(fluxes, material):
    if material == vps.Material.Mask:
        return 0.0
    rate = fluxes[1] * params["ionRate"]
    if material == vps.Material.Si:
        rate += fluxes[0] * params["neutralRate"]
    return rate


etchModel.setRateFunction(rateFunction)

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
