# This model/example currently only works in 2D mode
import viennaps2d as vps

vps.psLogger.setLogLevel(vps.psLogLevel.INTERMEDIATE)

# parse the parameters
params = vps.psReadConfigFile("config.txt")

stability = (
    2
    * params["diffusionCoefficient"]
    / max(params["scallopVelocity"], params["centerVelocity"])
)
print(f"Stability: {stability}")
if 0.5 * stability <= params["gridDelta"]:
    print("Unstable parameters. Reduce grid spacing!")

domain = vps.psDomain()
vps.psMakeStack(
    domain=domain,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=0.0,
    numLayers=int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius=params["trenchWidth"] / 2.0,
    maskHeight=0.0,
).apply()
# copy top layer for deposition
domain.duplicateTopLevelSet(vps.psMaterial.Polymer)

domain.generateCellSet(
    params["substrateHeight"] + params["numLayers"] * params["layerHeight"] + 10.0, True
)
cellSet = domain.getCellSet()
cellSet.addScalarData("byproductSum", 0.0)
cellSet.writeVTU("initial.vtu")
# we need neighborhood information for solving the
# convection-diffusion equation on the cell set
cellSet.buildNeighborhood()

# The redeposition model captures byproducts from the selective etching
# process in the cell set. The byproducts are then distributed by solving a
# convection-diffusion equation on the cell set.
model = vps.OxideRegrowthModel(
    params["nitrideEtchRate"] / 60.0,
    params["oxideEtchRate"] / 60.0,
    params["redepositionRate"],
    params["redepositionThreshold"],
    params["redepositionTimeInt"],
    params["diffusionCoefficient"],
    params["sink"],
    params["scallopVelocity"],
    params["centerVelocity"],
    params["substrateHeight"] + params["numLayers"] * params["layerHeight"],
    params["trenchWidth"],
)

process = vps.psProcess()
process.setDomain(domain)
process.setProcessModel(model)
process.setProcessDuration(params["targetEtchDepth"] / params["nitrideEtchRate"] * 60.0)
process.setPrintTimeInterval(30.0)

process.apply()

vps.psWriteVisualizationMesh(domain, "FinalStack").apply()
