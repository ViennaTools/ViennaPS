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

if args.dim == 2:
    print("Running 2D simulation.")
    ps.setDimension(2)
else:
    print("Running 3D simulation.")
    ps.setDimension(3)

params = ps.readConfigFile(args.filename)

geometry = ps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
ps.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    trenchTaperAngle=params["taperAngle"],
).apply()

geometry.duplicateTopLevelSet(ps.Material.SiO2)

model = ps.SingleParticleProcess(
    stickingProbability=params["stickingProbability"],
    sourceExponent=params["sourcePower"],
)

geometry.saveHullMesh("initial")

ps.Process(geometry, model, params["processTime"]).apply()

geometry.saveHullMesh("final")
