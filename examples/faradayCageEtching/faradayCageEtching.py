from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="faradayCageEtching", description="Run a faraday cage etching process."
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps as ps

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    psd = ps.d2
else:
    print("Running 3D simulation.")
    psd = ps.d3

params = ps.ReadConfigFile(args.filename)

# print intermediate output surfaces during the process
ps.Logger.setLogLevel(ps.LogLevel.INFO)
ps.setNumThreads(16)

# geometry setup, all units in um
geometry = psd.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    boundary=ps.BoundaryType.PERIODIC_BOUNDARY,
)
psd.MakeFin(
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

model = psd.FaradayCageEtching(mask, parameters)

# process setup
process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["etchTime"])  # seconds

# print initial surface
geometry.saveHullMesh(filename="initial")

# run the process
process.apply()

# print final surface
geometry.saveHullMesh(filename="final")
