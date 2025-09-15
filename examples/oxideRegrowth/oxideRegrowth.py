import sys
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="oxideRegrowth",
    description="Model oxide regrowth during SiN etching in SiN/SiO2 stack.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps as ps

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps.d2 as psd
else:
    print("Running 3D simulation.")
    import viennaps.d3 as psd

params = ps.readConfigFile(args.filename)

NUM_THREADS = 12
TIME_STABILITY_FACTOR = 0.245 if args.dim == 2 else 0.145

ps.Logger.setLogLevel(ps.LogLevel.INTERMEDIATE)

# Check for config file argument
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <config file>")
    sys.exit(1)

# Type alias for clarity
NumericType = float

# Calculate stability factor
diff_coeff = params["diffusionCoefficient"]
center_velocity = params["centerVelocity"]
scallop_velocity = params["scallopVelocity"]

stability = 2 * diff_coeff / max(center_velocity, scallop_velocity)
print(f"Stability: {stability}")

if 0.5 * stability <= params["gridDelta"]:
    print("Unstable parameters. Reduce grid spacing!")
    sys.exit(-1)

# Create domain
geometry = psd.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)

# Create stack geometry
psd.MakeStack(
    domain=geometry,
    numLayers=int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius=0.0,  # holeRadius
    trenchWidth=params["trenchWidth"],
    maskHeight=0.0,
).apply()

# Duplicate top layer
geometry.duplicateTopLevelSet(ps.Material.Polymer)

# Generate cell set above surface
geometry.generateCellSet(
    params["substrateHeight"] + params["numLayers"] * params["layerHeight"] + 10.0,
    ps.Material.GAS,
    True,
)

cell_set = geometry.getCellSet()
print("Cell set size: ", cell_set.getNumberOfCells())
cell_set.addScalarData("byproductSum", 0.0)
print("Added byproductSum")
cell_set.writeVTU("initial.vtu")

# Set periodic boundary (only relevant in 3D)
if args.dim == 3:
    boundary_conds = [False] * args.dim
    boundary_conds[1] = True
    cell_set.setPeriodicBoundary(boundary_conds)

# Build neighborhood for convection-diffusion
cell_set.buildNeighborhood()

# Create redeposition model
model = psd.OxideRegrowth(
    params["nitrideEtchRate"] / 60.0,
    params["oxideEtchRate"] / 60.0,
    params["redepositionRate"],
    params["redepositionThreshold"],
    params["redepositionTimeInt"],
    diff_coeff,
    params["sink"],
    scallop_velocity,
    center_velocity,
    params["substrateHeight"] + params["numLayers"] * params["layerHeight"],
    params["trenchWidth"],
    TIME_STABILITY_FACTOR,
)

# Run process
process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["targetEtchDepth"] / params["nitrideEtchRate"] * 60.0)
process.apply()

# Save output mesh
geometry.saveVolumeMesh("finalStack")
