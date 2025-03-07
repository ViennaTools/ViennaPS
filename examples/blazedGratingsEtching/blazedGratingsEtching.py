import viennaps2d as vps
import viennals2d as vls
import numpy as np
from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="blazedGratingsEtching",
    description="Run a oblique ion beam etching process on a grating.",
)
parser.add_argument("filename")
args = parser.parse_args()

vps.Logger.setLogLevel(vps.LogLevel.INFO)
params = vps.ReadConfigFile(args.filename)

########################################################################
#                               WARNING                                #
# This example in Python only works when using a single thread, i.e.   #
# vps.setNumThreads(1). The reason is the yield function, which is     #
# implemented in Python, but gets called from the C++ code by multiple #
# multiple threads. The Python GIL (Global Interpreter Lock) prevents  #
# the yield function from being called by multiple threads at the same #
# time, leading to deadlocked threads.                                 #
# It is therefore recommended to use the C++ API for this example.     #
########################################################################
vps.setNumThreads(1)

# ----- Geometry Generation ----- #
bumpWidth = params["bumpWidth"]
bumpHeight = params["bumpHeight"]
bumpDuty = params["bumpDuty"]
numBumps = int(params["numBumps"])
bumpSpacing = bumpWidth * (1.0 - bumpDuty) / bumpDuty
xExtent = numBumps * bumpWidth / bumpDuty

geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=xExtent,
    boundary=vps.BoundaryType.PERIODIC_BOUNDARY,
)
vps.MakePlane(domain=geometry, height=0.0, material=vps.Material.SiO2).apply()
mask = vls.Domain(geometry.getGrid())

mesh = vls.Mesh()
offset = -xExtent / 2.0 + bumpSpacing + bumpWidth / 2.0
numNodes = 100
for i in range(numNodes):
    x = -bumpWidth / 2.0 + i * bumpWidth / (numNodes - 1)
    y = -4.0 * bumpHeight * x * x / (bumpWidth * bumpWidth) + bumpHeight
    mesh.insertNextNode([x + offset, y, 0.0])
for i in range(1, numNodes):
    mesh.insertNextLine([i - 1, i])
mesh.insertNextLine([numNodes - 1, 0])

for i in range(numBumps):
    tip = vls.Domain(geometry.getGrid())
    vls.FromSurfaceMesh(tip, mesh).apply()
    vls.TransformMesh(
        mesh=mesh,
        transform=vls.TransformEnum.TRANSLATION,
        transformVector=[bumpSpacing + bumpWidth, 0, 0],
    ).apply()
    vls.BooleanOperation(mask, tip, vls.BooleanOperationEnum.UNION).apply()

geometry.insertNextLevelSetAsMaterial(mask, vps.Material.Mask)
geometry.saveSurfaceMesh("initial", True)

# ----- Model Setup ----- #
advectionParams = vps.AdvectionParameters()
advectionParams.integrationScheme = vls.IntegrationSchemeEnum.LAX_FRIEDRICHS_2ND_ORDER
advectionParams.timeStepRatio = 0.25

rayTracingParams = vps.RayTracingParameters()
rayTracingParams.raysPerPoint = int(params["raysPerPoint"])
rayTracingParams.smoothingNeighbors = 1

yieldFactor = params["yieldFactor"]


def yieldFunction(theta):
    cosTheta = np.cos(theta)
    return (
        yieldFactor * cosTheta
        - 1.55 * cosTheta * cosTheta
        + 0.65 * cosTheta * cosTheta * cosTheta
    ) / (yieldFactor - 0.9)


ibeParams = vps.IBEParameters()
ibeParams.exponent = params["exponent"]
ibeParams.meanEnergy = params["meanEnergy"]
ibeParams.materialPlaneWaferRate = {vps.Material.SiO2: 1, vps.Material.Mask: 1 / 11}
ibeParams.yieldFunction = yieldFunction

model = vps.IonBeamEtching()

# ----- ANSGM Etch ----- #
angle = params["phi1"]
direction = [0.0, 0.0, 0.0]
direction[0] = -np.sin(np.deg2rad(angle))
direction[1] = -np.cos(np.deg2rad(angle))
ibeParams.tiltAngle = angle
model.setPrimaryDirection(direction)
model.setParameters(ibeParams)

process = vps.Process(geometry, model, 0.0)
process.setAdvectionParameters(advectionParams)
process.setRayTracingParameters(rayTracingParams)

process.setProcessDuration(params["ANSGM_Depth"])
process.apply()
geometry.saveSurfaceMesh("ANSGM_Etch", True)

# remove mask
geometry.removeTopLevelSet()
geometry.saveSurfaceMesh("ANSGM", True)

# ------ Blazed Gratins Etch ------ #
angle = params["phi2"]
direction[0] = -np.sin(np.deg2rad(angle))
direction[1] = -np.cos(np.deg2rad(angle))
ibeParams.tiltAngle = angle
model.setPrimaryDirection(direction)
model.setParameters(ibeParams)

for i in range(1, 5):
    process.setProcessDuration(params["etchTimeP" + str(i)])
    process.apply()
    geometry.saveSurfaceMesh("BlazedGratingsEtch_P" + str(i))
