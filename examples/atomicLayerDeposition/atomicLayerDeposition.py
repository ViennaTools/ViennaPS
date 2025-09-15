import viennaps as ps
import viennals as ls


params = ps.readConfigFile("config.txt")
geometry = ps.d2.Domain()

# Create the geometry
boundaryCons = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]
gridDelta = params["gridDelta"]
bounds = [
    0.0,
    params["openingWidth"] / 2.0 + params["xPad"] + params["gapLength"],
    -gridDelta,
    params["openingDepth"] + params["gapHeight"] + gridDelta,
]

substrate = ls.d2.Domain(bounds, boundaryCons, gridDelta)
normal = [0.0, 1.0]
origin = [0.0, params["openingDepth"] + params["gapHeight"]]
ls.d2.MakeGeometry(substrate, ls.d2.Plane(origin, normal)).apply()

geometry.insertNextLevelSetAsMaterial(substrate, ps.Material.Si)

vertBox = ls.d2.Domain(bounds, boundaryCons, gridDelta)
minPoint = [-gridDelta, 0.0]
maxPoint = [
    params["openingWidth"] / 2.0,
    params["gapHeight"] + params["openingDepth"] + gridDelta,
]
ls.d2.MakeGeometry(vertBox, ls.d2.Box(minPoint, maxPoint)).apply()

geometry.applyBooleanOperation(vertBox, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

horiBox = ls.d2.Domain(bounds, boundaryCons, gridDelta)
minPoint = [params["openingWidth"] / 2.0 - gridDelta, 0.0]
maxPoint = [params["openingWidth"] / 2.0 + params["gapLength"], params["gapHeight"]]
ls.d2.MakeGeometry(horiBox, ls.d2.Box(minPoint, maxPoint)).apply()
geometry.applyBooleanOperation(horiBox, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

geometry.saveVolumeMesh("SingleParticleALD_initial.vtu")

geometry.duplicateTopLevelSet(ps.Material.Al2O3)

gasMFP = ps.constants.gasMeanFreePath(
    params["pressure"], params["temperature"], params["diameter"]
)
print("Mean free path: ", gasMFP, " um")

model = ps.d2.SingleParticleALD(
    stickingProbability=params["stickingProbability"],
    numCycles=int(params["numCycles"]),
    growthPerCycle=params["growthPerCycle"],
    totalCycles=int(params["totalCycles"]),
    coverageTimeStep=params["coverageTimeStep"],
    evFlux=params["evFlux"],
    inFlux=params["inFlux"],
    s0=params["s0"],
    gasMFP=gasMFP,
)

alpParams = ps.AtomicLayerProcessParameters()
alpParams.pulseTime = params["pulseTime"]
alpParams.coverageTimeStep = params["coverageTimeStep"]
alpParams.numCycles = int(params["numCycles"])

rayParams = ps.RayTracingParameters()
rayParams.raysPerPoint = int(params["numRaysPerPoint"])

ALP = ps.d2.Process(geometry, model)
ALP.setRayTracingParameters(rayParams)
ALP.setAtomicLayerProcessParameters(alpParams)
ALP.apply()

## TODO: Implement MeasureProfile in Python
#   MeasureProfile<NumericType, D>(domain, params.get("gapHeight") / 2.)
#       .save(params.get<std::string>("outputFile"));

geometry.saveVolumeMesh("SingleParticleALD_final.vtu")
