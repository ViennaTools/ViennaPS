from argparse import ArgumentParser

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
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

params = vps.ReadConfigFile(args.filename)

# print intermediate output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)

# geometry setup, all units in um
geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    boundary=vps.BoundaryType.PERIODIC_BOUNDARY,
)
vps.MakeFin(
    domain=geometry,
    finWidth=params["finWidth"],
    finHeight=0.0,
    maskHeight=params["maskHeight"],
).apply()

# use pre-defined etching model
parameters = vps.FaradayCageParameters()
parameters.cageAngle = params["cageAngle"]
parameters.ibeParams.tiltAngle = params["tiltAngle"]
mask = [vps.Material.Mask]

model = vps.FaradayCageEtching(mask, parameters)

# process setup
process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setNumberOfRaysPerPoint(int(params["raysPerPoint"]))
process.setProcessDuration(params["etchTime"])  # seconds

# print initial surface
geometry.saveHullMesh(filename="initial")

# run the process
process.apply()

# print final surface
geometry.saveHullMesh(filename="final")
