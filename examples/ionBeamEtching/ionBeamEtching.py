from argparse import ArgumentParser
import numpy as np
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="ionBeamEtching",
    description="Run an IBE process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    ps.setDimension(2)
else:
    print("Running 3D simulation.")
    ps.setDimension(3)

ps.Logger.setLogLevel(ps.LogLevel.INTERMEDIATE)

params = ps.readConfigFile(args.filename)

geometry = ps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
ps.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchDepth"],
    maskHeight=params["maskHeight"],
).apply()

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(ps.Material.Polymer)

ibeParams = ps.IBEParameters()
ibeParams.tiltAngle = params["angle"]
ibeParams.exponent = params["exponent"]

ibeParams.meanEnergy = params["meanEnergy"]
ibeParams.sigmaEnergy = params["sigmaEnergy"]
ibeParams.thresholdEnergy = params["thresholdEnergy"]

ibeParams.redepositionRate = params["redepositionRate"]
ibeParams.planeWaferRate = params["planeWaferRate"]

model = ps.IonBeamEtching(
    maskMaterials=[ps.Material.Mask],
    parameters=ibeParams,
)

direction = [0.0, 0.0, 0.0]
direction[args.dim - 1] = -np.cos(ibeParams.tiltAngle * np.pi / 180.0)
direction[args.dim - 2] = np.sin(ibeParams.tiltAngle * np.pi / 180.0)
model.setPrimaryDirection(direction)

advParams = ps.AdvectionParameters()
advParams.integrationScheme = ps.IntegrationScheme.LAX_FRIEDRICHS_2ND_ORDER

process = ps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.setAdvectionParameters(advParams)

geometry.saveHullMesh("initial")

process.apply()

geometry.saveHullMesh("final")
