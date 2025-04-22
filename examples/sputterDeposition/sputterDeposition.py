from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser(
    prog="trenchDeposition",
    description="Run trench deposition with a rate profile from a CSV file.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
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

# Setup
vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vps.setNumThreads(16)

# Load parameters
params = vps.ReadConfigFile(args.filename)

if args.visualize:
    from visualizeDomain import visualize2d
    visualize2d(
        rates_file=params["ratesFile"].strip(),
        offset=params["offsetX"],
        x_extent=params["xExtent"],
        trench_width=params["trenchWidth"],
    )

# Geometry setup
geometry = vps.Domain()
vps.MakeTrench(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchDepth"],
    taperingAngle=params["taperingAngle"],
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=vps.Material.Si,
).apply()

geometry.saveVolumeMesh("Trench")
geometry.duplicateTopLevelSet(vps.Material.SiO2)

# Setup direction
direction = [0.0, -1.0, 0.0]

# Setup offset
offset = [0.0, 0.0]
offset[0] = params["offsetX"]

# Create CSV-based deposition process
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

# Simulation loop
numCycles = int(params["numCycles"])
filename_prefix = "TrenchDeposition_"

n = 0
geometry.saveSurfaceMesh(f"{filename_prefix}{n}.vtp")
n += 1

for i in range(numCycles):
    print(f"Cycle {i + 1}")
    vps.Process(geometry, depoModel, params["depositionTime"]).apply()
    geometry.saveSurfaceMesh(f"{filename_prefix}{n}.vtp")
    n += 1

geometry.saveVolumeMesh("TrenchFinal")
