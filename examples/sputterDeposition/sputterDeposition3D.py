from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

# Argument parsing
parser = ArgumentParser(
    prog="holeDeposition",
    description="Run hole deposition with a CSV-defined rate profile.",
)
parser.add_argument("filename")
parser.add_argument("--plot-rates", action="store_true", help="Plot the deposition rate profile and hole domain")
args = parser.parse_args()

# Import 3D ViennaPS
import viennaps3d as vps

# Setup logging and threads
vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vps.setNumThreads(16)

# Load config
params = vps.ReadConfigFile(args.filename)

# Optional rate profile plot
if args.plot_rates:
    # Load rate CSV with header: assumes columns x,y,rate
    data = np.loadtxt(params["ratesFile"].strip(), delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    rate = data[:, 2]

    offset_x = params["offsetX"]
    offset_y = params["offsetY"]
    radius = params["holeRadius"]
    x_extent = params["xExtent"]
    y_extent = params["yExtent"]

    # Create a scatter plot (top-down view)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=rate, cmap="coolwarm", s=10)
    plt.colorbar(sc, label="Deposition Rate")

    # Draw extent box and hole area
    plt.gca().add_patch(
        plt.Rectangle(
            (offset_x - x_extent / 2, offset_y - y_extent / 2),
            x_extent,
            y_extent,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            linewidth=1.5,
            label="Simulation Domain"
        )
    )
    hole = plt.Circle(
        (offset_x, offset_y),
        radius,
        color="orange",
        alpha=0.3,
        label="Hole Area"
    )
    plt.gca().add_patch(hole)

    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    plt.title("Deposition Rate Map with Simulation Region")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rate_profile_3d.png", dpi=300)
    print("Saved 3D rate profile plot as 'rate_profile_3d.png'")

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
offset = [params["offsetX"], params["offsetY"]]

# CSV-based deposition model
depoModel = vps.CSVFileProcess(
    ratesFile=params["ratesFile"].strip(),
    direction=direction,
    offset=offset,
)

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
