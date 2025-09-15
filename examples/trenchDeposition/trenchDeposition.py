from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="trenchDeposition",
    description="Run a deposition process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
import viennaps

if args.dim == 2:
    print("Running 2D simulation.")
    psd = ps.d2
else:
    print("Running 3D simulation.")
    psd = ps.d3

params = ps.ReadConfigFile(args.filename)

geometry = psd.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
psd.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    trenchTaperAngle=params["taperAngle"],
).apply()

geometry.duplicateTopLevelSet(ps.Material.SiO2)

model = psd.SingleParticleProcess(
    stickingProbability=params["stickingProbability"],
    sourceExponent=params["sourcePower"],
)

geometry.saveHullMesh("initial")

psd.Process(geometry, model, params["processTime"]).apply()

geometry.saveHullMesh("final")
