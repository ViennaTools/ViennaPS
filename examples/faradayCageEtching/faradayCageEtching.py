from argparse import ArgumentParser
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="faradayCageEtching", description="Run a faraday cage etching process."
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()


# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
else:
    print("Running 3D simulation.")
ps.setDimension(args.dim)
params = ps.readConfigFile(args.filename)

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)
ps.setNumThreads(16)

# geometry setup, all units in um
geometry = ps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    boundary=ps.BoundaryType.PERIODIC_BOUNDARY,
)
ps.MakeFin(
    domain=geometry,
    finWidth=params["finWidth"],
    finHeight=0.0,
    maskHeight=params["maskHeight"],
).apply()

# use pre-defined etching model
parameters = ps.FaradayCageParameters()
parameters.cageAngle = params["cageAngle"]
parameters.ibeParams.tiltAngle = params["tiltAngle"]
mask = [ps.Material.Mask]

model = ps.FaradayCageEtching(parameters, mask)

advParams = ps.AdvectionParameters()
advParams.spatialScheme = ps.SpatialScheme.LAX_FRIEDRICHS_1ST_ORDER

# process setup
process = ps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setParameters(advParams)
process.setProcessDuration(params["etchTime"])  # seconds

# print initial surface
geometry.saveHullMesh(filename="initial")

# run the process
process.apply()

# print final surface
geometry.saveHullMesh(filename="final")
