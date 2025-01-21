from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(prog="faradayCageEtching", description="Run a faraday cage etching process.")
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
geometry = vps.Domain()
vps.MakeFin(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    finWidth=params["finWidth"],
    finHeight=params["maskHeight"],
    periodicBoundary=True,
    makeMask=True,
    material=vps.Material.Si,
).apply()

# use pre-defined model SF6O2 etching model
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
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

# run the process
process.apply()

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
