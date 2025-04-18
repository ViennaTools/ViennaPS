from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

# Argument parsing
parser = ArgumentParser(
    prog="trenchDeposition",
    description="Run trench deposition with a rate profile from a CSV file.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
parser.add_argument("--plot-rates", action="store_true", help="Plot the deposition rate profile and trench domain")
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

if args.plot_rates:
    # Load CSV and skip header row
    data = np.loadtxt(params["ratesFile"].strip(), delimiter=",", skiprows=1)
    x = data[:, 0]
    rate = data[:, 1]

    offset = params["offsetX"]
    x_extent = params["xExtent"]
    trench_width = params["trenchWidth"]

    domain_start = offset - x_extent / 2
    domain_end = offset + x_extent / 2
    trench_start = offset - trench_width / 2
    trench_end = offset + trench_width / 2

    plt.figure(figsize=(10, 5))
    plt.plot(x, rate, label="Deposition Rate", linewidth=2)

    plt.axvspan(domain_start, domain_end, color="lightblue", alpha=0.3, label="Simulation Domain")
    plt.axvspan(trench_start, trench_end, color="orange", alpha=0.4, label="Trench Area")
    plt.axvline(offset, color="black", linestyle="--", alpha=0.5, label="Offset X")

    plt.xlabel("x [μm]")
    plt.ylabel("Deposition Rate")
    plt.title("Deposition Rate Profile with Simulation Region")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rate_profile.png", dpi=300)
    print("Saved rate profile plot as 'rate_profile.png'")

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
offset = [params["offsetX"]]

# Create CSV-based deposition process
depoModel = vps.CSVFileProcess(
    ratesFile=params["ratesFile"].strip(),
    direction=direction,
    offset=offset,
)

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
