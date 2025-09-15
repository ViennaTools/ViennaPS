from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="multiTEOS",
    description="Run a multi TEOS deposition process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
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


# copy top layer to capture deposition
geometry.duplicateTopLevelSet(ps.Material.SiO2)

# process model encompasses surface model and particle types
model = psd.TEOSDeposition(
    stickingProbabilityP1=params["stickingProbabilityP1"],
    rateP1=params["depositionRateP1"],
    orderP1=params["reactionOrderP1"],
)

rayParams = ps.RayTracingParameters()
rayParams.raysPerPoint = int(params["numRaysPerPoint"])

process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setRayTracingParameters(rayParams)
process.setProcessDuration(params["processTime"])

geometry.saveVolumeMesh("SingleTEOS_initial.vtp")

process.apply()

geometry.saveVolumeMesh("SingleTEOS_final.vtp")
