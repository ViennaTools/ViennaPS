import viennaps2d as vps
import viennals2d as vls  


params = vps.ReadConfigFile("config.txt")
geometry = vps.Domain()

# Create the geometry
boundaryCons = [vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY, vls.BoundaryConditionEnum.INFINITE_BOUNDARY]
gridDelta = params["gridDelta"]
bounds = [0., 
          params["openingWidth"] / 2. + params["xPad"] + params["gapLength"], 
          -gridDelta,
          params["openingDepth"] + params["gapHeight"] + gridDelta]

substrate = vls.Domain(bounds, boundaryCons, gridDelta)
normal = [0., 1.]
origin = [0., params["openingDepth"] + params["gapHeight"]]
vls.MakeGeometry(substrate, vls.Plane(origin, normal)).apply()

geometry.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)

vertBox = vls.Domain(bounds, boundaryCons, gridDelta)
minPoint = [-gridDelta, 0.]
maxPoint = [params["openingWidth"] / 2., params["gapHeight"] + params["openingDepth"] + gridDelta]
vls.MakeGeometry(vertBox, vls.Box(minPoint, maxPoint)).apply()

geometry.applyBooleanOperation(vertBox, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

horiBox = vls.Domain(bounds, boundaryCons, gridDelta)
minPoint = [params["openingWidth"] / 2. - gridDelta, 0.]
maxPoint = [params["openingWidth"] / 2. + params["gapLength"], params["gapHeight"]]
vls.MakeGeometry(horiBox, vls.Box(minPoint, maxPoint)).apply()
geometry.applyBooleanOperation(horiBox, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

geometry.saveVolumeMesh("SingleParticleALD_initial.vtu")

geometry.duplicateTopLevelSet(vps.Material.Al2O3)

gasMFP = vps.constants.gasMeanFreePath(params["pressure"], params["temperature"], params["diameter"])
print("Mean free path: ", gasMFP, " um")

model = vps.SingleParticleALD(params["stickingProbability"], params["numCycles"],
                                params["growthPerCycle"], params["totalCycles"],
                                params["coverageTimeStep"], params["evFlux"],
                                params["inFlux"], params["s0"], gasMFP)

ALP = vps.AtomicLayerProcess(geometry, model)
ALP.setCoverageTimeStep(params["coverageTimeStep"])
ALP.setPulseTime(params["pulseTime"])
ALP.setNumCycles(int(params["numCycles"]))
ALP.setNumberOfRaysPerPoint(int(params["numRaysPerPoint"]))
ALP.disableRandomSeeds()
ALP.apply()

## TODO: Implement MeasureProfile in Python
#   MeasureProfile<NumericType, D>(domain, params.get("gapHeight") / 2.)
#       .save(params.get<std::string>("outputFile"));

geometry.saveVolumeMesh("SingleParticleALD_final.vtu")