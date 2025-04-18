from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser(
    prog="holeDeposition",
    description="Run hole deposition with a CSV-defined rate profile.",
)
parser.add_argument("filename")
args = parser.parse_args()

# Import 3D ViennaPS
import viennaps3d as vps

# Setup logging and threads
vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vps.setNumThreads(16)

# Load config
params = vps.ReadConfigFile(args.filename)

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
