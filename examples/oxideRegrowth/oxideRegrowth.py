import viennacs2d
import sys
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="faradayCageEtching", description="Run a faraday cage etching process."
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

NUM_THREADS = 12
TIME_STABILITY_FACTOR = 0.245 if args.dim == 2 else 0.145

vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)

# Check for config file argument
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <config file>")
    sys.exit(1)

# Type alias for clarity
NumericType = float

# Calculate stability factor
diff_coeff = params.get("diffusionCoefficient")
center_velocity = params.get("centerVelocity")
scallop_velocity = params.get("scallopVelocity")

stability = 2 * diff_coeff / max(center_velocity, scallop_velocity)
print(f"Stability: {stability}")

if 0.5 * stability <= params.get("gridDelta"):
    print("Unstable parameters. Reduce grid spacing!")
    sys.exit(-1)

# Create domain
geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)

# Create stack geometry
vps.MakeStack(
    domain = geometry,
    numLayers = int(params["numLayers"]),
    layerHeight=params["layerHeight"],
    substrateHeight=params["substrateHeight"],
    holeRadius = 0.0,  # holeRadius
    trenchWidth = params["trenchWidth"],
    maskHeight=0.0,
).apply()

# Duplicate top layer
geometry.duplicateTopLevelSet(vps.Material.Polymer)

# Generate cell set above surface
geometry.generateCellSet(
    params.get("substrateHeight") + params.get("numLayers") * params.get("layerHeight") + 10.0,
    vps.Material.GAS,
    True
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
model = vps.OxideRegrowth(
    params.get("nitrideEtchRate") / 60.0,
    params.get("oxideEtchRate") / 60.0,
    params.get("redepositionRate"),
    params.get("redepositionThreshold"),
    params.get("redepositionTimeInt"),
    diff_coeff,
    params.get("sink"),
    scallop_velocity,
    center_velocity,
    params.get("substrateHeight") + params.get("numLayers") * params.get("layerHeight"),
    params.get("trenchWidth"),
    TIME_STABILITY_FACTOR
)

# Run process
process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params.get("targetEtchDepth") / params.get("nitrideEtchRate") * 60.0)
process.apply()

# Save output mesh
geometry.saveVolumeMesh("finalStack")
