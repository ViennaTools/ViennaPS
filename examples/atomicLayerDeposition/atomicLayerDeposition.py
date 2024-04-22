import viennaps2d as vps


def makeLShape(params: dict, material: vps.Material) -> vps.Domain:
    try:
        import viennals2d as vls
    except ImportError:
        print(
            "The ViennaLS Python library is required to run this example. "
            "Please install it and try again."
        )
        return

    domain = vps.Domain()
    gridDelta = params["gridDelta"]
    bounds = [0] * vps.D * 2
    bounds[0] = -params["verticalWidth"] / 2.0 - params["xPad"]
    bounds[1] = params["verticalWidth"] / 2.0 + params["xPad"]
    if vps.D == 3:
        bounds[2] = -params["verticalWidth"] / 2.0 - params["xPad"]
        bounds[3] = (
            -params["verticalWidth"] / 2.0 + params["xPad"] + params["horizontalWidth"]
        )
    else:
        bounds[1] = (
            -params["verticalWidth"] / 2.0 + params["xPad"] + params["horizontalWidth"]
        )

    boundaryCons = [vls.lsBoundaryConditionEnum.REFLECTIVE_BOUNDARY] * (vps.D - 1)
    boundaryCons.append(vls.lsBoundaryConditionEnum.INFINITE_BOUNDARY)

    substrate = vls.lsDomain(bounds, boundaryCons, gridDelta)
    normal = [0.0] * vps.D
    origin = [0.0] * vps.D
    normal[vps.D - 1] = 1.0
    origin[vps.D - 1] = params["verticalDepth"]
    vls.lsMakeGeometry(substrate, vls.lsPlane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(substrate, material)

    # Create the vertical trench
    vertBox = vls.lsDomain(domain.getLevelSets()[0])
    minPoint = [0] * vps.D
    maxPoint = [0] * vps.D
    for i in range(vps.D - 1):
        minPoint[i] = -params["verticalWidth"] / 2.0
        maxPoint[i] = params["verticalWidth"] / 2.0
    maxPoint[vps.D - 1] = params["verticalDepth"]
    vls.lsMakeGeometry(vertBox, vls.lsBox(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(
        vertBox, vls.lsBooleanOperationEnum.RELATIVE_COMPLEMENT
    )

    # Create the horizontal trench
    horiBox = vls.lsDomain(domain.getLevelSets()[0])
    minPoint = [0] * vps.D
    maxPoint = [params["verticalWidth"] / 2.0] * vps.D
    for i in range(vps.D - 1):
        minPoint[i] = -params["verticalWidth"] / 2.0
    maxPoint[vps.D - 1] = params["horizontalHeight"]
    maxPoint[vps.D - 2] = -params["verticalWidth"] / 2.0 + params["horizontalWidth"]
    vls.lsMakeGeometry(horiBox, vls.lsBox(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(
        horiBox, vls.lsBooleanOperationEnum.RELATIVE_COMPLEMENT
    )
    return domain


vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)
params = vps.ReadConfigFile("config.txt")
vps.setNumThreads(int(params["numThreads"]))

# Create a domain
domain = makeLShape(params, vps.Material.TiN)
domain.generateCellSet(
    params["verticalDepth"] + params["topSpace"], vps.Material.GAS, True
)
cellSet = domain.getCellSet()

# Segment the cells into surface, material, and gas cells
vps.SegmentCells(cellSet).apply()
cellSet.writeVTU("initial.vtu")

timer = vps.Timer()
timer.start()

# Calculate the mean free path for the gas cells
mfpCalc = vps.MeanFreePath()
mfpCalc.setDomain(domain)
mfpCalc.setNumRaysPerCell(params["raysPerCell"])
mfpCalc.setReflectionLimit(int(params["reflectionLimit"]))
mfpCalc.setRngSeed(int(params["seed"]))
mfpCalc.setMaterial(vps.Material.GAS)
mfpCalc.setBulkLambda(params["bulkLambda"])
mfpCalc.apply()

timer.finish()
print(f"Mean free path calculation took {timer.totalDuration * 1e-9} seconds.")

# Run the atomic layer deposition model
model = vps.csAtomicLayerProcess(domain)
model.setMaxLambda(params["bulkLambda"])
model.setPrintInterval(params["printInterval"])
model.setStabilityFactor(params["stabilityFactor"])
model.setFirstPrecursor(
    "H2O",
    params["H2O_meanThermalVelocity"],
    params["H2O_adsorptionRate"],
    params["H2O_desorptionRate"],
    params["p1_time"],
    params["inFlux"],
)
model.setSecondPrecursor(
    "TMA",
    params["TMA_meanThermalVelocity"],
    params["TMA_adsorptionRate"],
    params["TMA_desorptionRate"],
    params["p2_time"],
    params["inFlux"],
)
model.setPurgeParameters(params["purge_meanThermalVelocity"], params["purge_time"])
model.setReactionOrder(params["reactionOrder"])
model.apply()
