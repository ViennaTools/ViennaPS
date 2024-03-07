from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="oxideRegrowth", description="Run an oxide regrowth process."
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

if args.dim == 3:
    print("Running 3D simulation.")
    stabFac = 0.145
    import viennaps3d as vps
else:
    print("Running 2D simulation.")
    stabFac = 0.245
    import viennaps2d as vps

# print intermediate output volumes during the process
vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)

params = vps.ReadConfigFile(args.filename)

stability = (
    2
    * params["diffusionCoefficient"]
    / max(params["scallopVelocity"], params["centerVelocity"])
)
print(f"Stability: {stability}")
if 0.5 * stability <= params["gridDelta"]:
    print("Unstable parameters. Reduce grid spacing!")

domain = vps.Domain()
vps.MakeStack(
    domain=domain,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    numLayers=int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius=0.0,
    trenchWidth=params["trenchWidth"],
    maskHeight=0.0,
).apply()
# copy top layer for deposition
domain.duplicateTopLevelSet(vps.Material.Polymer)

domain.generateCellSet(
    params["substrateHeight"] + params["numLayers"] * params["layerHeight"] + 10.0, True
)
cellSet = domain.getCellSet()
cellSet.addScalarData("byproductSum", 0.0)
cellSet.writeVTU("initial.vtu")
if args.dim == 3:
    cellSet.setPeriodicBoundary([False, True, False])
# we need neighborhood information for solving the
# convection-diffusion equation on the cell set
cellSet.buildNeighborhood()


# The redeposition model captures byproducts from the selective etching
# process in the cell set. The byproducts are then distributed by solving a
# convection-diffusion equation on the cell set.
model = vps.OxideRegrowth(
    nitrideEtchRate=params["nitrideEtchRate"] / 60.0,
    oxideEtchRate=params["oxideEtchRate"] / 60.0,
    redepositionRate=params["redepositionRate"],
    redepositionThreshold=params["redepositionThreshold"],
    redepositionTimeInt=params["redepositionTimeInt"],
    diffusionCoefficient=params["diffusionCoefficient"],
    sinkStrength=params["sink"],
    scallopVelocity=params["scallopVelocity"],
    centerVelocity=params["centerVelocity"],
    topHeight=params["substrateHeight"] + params["numLayers"] * params["layerHeight"],
    centerWidth=params["trenchWidth"],
    stabilityFactor=stabFac,
)

process = vps.Process()
process.setDomain(domain)
process.setProcessModel(model)
process.setProcessDuration(params["targetEtchDepth"] / params["nitrideEtchRate"] * 60.0)
process.setPrintTimeInterval(30.0)

process.apply()

domain.saveVolumeMesh("finalStack")
