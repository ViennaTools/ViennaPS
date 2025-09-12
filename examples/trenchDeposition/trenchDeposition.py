from argparse import ArgumentParser

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
    import viennaps.d2 as vps
else:
    print("Running 3D simulation.")
    import viennaps.d3 as vps

params = viennaps.ReadConfigFile(args.filename)

geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
vps.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    trenchTaperAngle=params["taperAngle"],
).apply()

geometry.duplicateTopLevelSet(viennaps.Material.SiO2)

model = vps.SingleParticleProcess(
    stickingProbability=params["stickingProbability"],
    sourceExponent=params["sourcePower"],
)

geometry.saveHullMesh("initial")

vps.Process(geometry, model, params["processTime"]).apply()

geometry.saveHullMesh("final")
