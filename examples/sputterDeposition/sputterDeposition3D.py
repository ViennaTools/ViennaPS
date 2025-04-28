from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser(
    prog="holeDeposition",
    description="Run hole deposition with a CSV-defined rate profile.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=3)
parser.add_argument("filename")
parser.add_argument("--visualize", action="store_true", help="Visualize the rate and geometry domains")
args = parser.parse_args()

# Select dimension module
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

# Setup logging and threads
vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vps.setNumThreads(16)

# Load config
params = vps.ReadConfigFile(args.filename)

# Optional rate profile plot
if args.visualize:
    from visualizeDomain import visualize3d
    visualize3d(
        rates_file=params["ratesFile"].strip(),
        offset_x=params["offsetX"],
        offset_y=params["offsetY"],
        radius=params["holeRadius"],
        x_extent=params["xExtent"],
        y_extent=params["yExtent"],
    )

# Geometry setup
geometry = vps.Domain()
vps.MakeHole(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    holeRadius=params["holeRadius"],
    holeDepth=params["holeDepth"],
    taperingAngle=params["taperingAngle"],
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=vps.Material.Si,
    holeShape=vps.HoleShape.Full,
).apply()

geometry.saveVolumeMesh("Hole")
geometry.duplicateTopLevelSet(vps.Material.SiO2)

# Direction: vertical deposition
direction = [0.0, 0.0, -1.0]

# Offset in X, Y
offset = [0.0, 0.0]
offset[0] = params["offsetX"]
offset[1] = params["offsetY"]

# CSV-based deposition model
depoModel = vps.CSVFileProcess(
    ratesFile=params["ratesFile"].strip(),
    direction=direction,
    offset=offset,
)

# Select interpolation mode
mode = params["interpolationMode"].strip().lower()
depoModel.setInterpolationMode(mode)
if mode == "custom":
    def custom_interp(coord):
        import numpy as np
        x, _, _ = coord
        return 0.04 + 0.01 * np.sin(10.0 * x)
    depoModel.setCustomInterpolator(custom_interp)

# Simulation
numCycles = int(params["numCycles"])
filename_prefix = "HoleDeposition_"

n = 0
geometry.saveSurfaceMesh(f"{filename_prefix}{n}.vtp")
n += 1

for i in range(numCycles):
    print(f"Cycle {i + 1}")
    vps.Process(geometry, depoModel, params["depositionTime"]).apply()
    geometry.saveSurfaceMesh(f"{filename_prefix}{n}.vtp")
    n += 1

geometry.saveVolumeMesh("HoleFinal")
