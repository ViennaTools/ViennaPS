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

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(ps.Material.SiO2)

# process model encompasses surface model and particle types
model = ps.TEOSDeposition(
    stickingProbabilityP1=params["stickingProbabilityP1"],
    rateP1=params["depositionRateP1"],
    orderP1=params["reactionOrderP1"],
    stickingProbabilityP2=params["stickingProbabilityP2"],
    rateP2=params["depositionRateP2"],
    orderP2=params["reactionOrderP2"],
)

rayParams = ps.RayTracingParameters()
rayParams.raysPerPoint = int(params["numRaysPerPoint"])

process = ps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setRayTracingParameters(rayParams)
process.setProcessDuration(params["processTime"])

geometry.saveVolumeMesh("MultiTEOS_initial.vtp")

process.apply()

geometry.saveVolumeMesh("MultiTEOS_final.vtp")
