import viennaps as ps
import viennals as ls
import matplotlib.pyplot as plt
import numpy as np

ps.setDimension(2)
ls.setDimension(2)

ps.Logger.setLogLevel(ps.LogLevel.DEBUG)


def run_simulation(gd, filename):
    params = ps.readConfigFile("config.txt")
    geometry = ps.Domain()

    # Create the geometry
    boundaryCons = [
        ps.BoundaryType.REFLECTIVE_BOUNDARY,
        ps.BoundaryType.INFINITE_BOUNDARY,
    ]
    gridDelta = gd
    bounds = [
        0.0,
        params["openingWidth"] / 2.0 + params["xPad"] + params["gapLength"],
        -gridDelta,
        params["openingDepth"] + params["gapHeight"] + gridDelta,
    ]

    substrate = ls.Domain(bounds, boundaryCons, gridDelta)
    normal = [0.0, 1.0]
    origin = [0.0, params["openingDepth"] + params["gapHeight"]]
    ls.MakeGeometry(substrate, ls.Plane(origin, normal)).apply()

    geometry.insertNextLevelSetAsMaterial(substrate, ps.Material.Si)

    vertBox = ls.Domain(bounds, boundaryCons, gridDelta)
    minPoint = [-gridDelta, 0.0]
    maxPoint = [
        params["openingWidth"] / 2.0,
        params["gapHeight"] + params["openingDepth"] + gridDelta,
    ]
    ls.MakeGeometry(vertBox, ls.Box(minPoint, maxPoint)).apply()

    geometry.applyBooleanOperation(vertBox, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    horiBox = ls.Domain(bounds, boundaryCons, gridDelta)
    minPoint = [params["openingWidth"] / 2.0 - gridDelta, 0.0]
    maxPoint = [params["openingWidth"] / 2.0 + params["gapLength"], params["gapHeight"]]
    ls.MakeGeometry(horiBox, ls.Box(minPoint, maxPoint)).apply()
    geometry.applyBooleanOperation(horiBox, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    geometry.saveSurfaceMesh("SingleParticleALD_initial")

    model = ps.SingleParticleProcess(rate=1.0, stickingProbability=1e-4)

    geometry.duplicateTopLevelSet(ps.Material.Al2O3)

    gasMFP = ps.constants.gasMeanFreePath(
        params["pressure"], params["temperature"], params["diameter"]
    )
    print("Mean free path: ", gasMFP, " um")

    model = ps.SingleParticleALD(
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
    model.setProcessName(filename)

    alpParams = ps.AtomicLayerProcessParameters()
    alpParams.pulseTime = params["pulseTime"]
    alpParams.coverageTimeStep = params["coverageTimeStep"]
    alpParams.numCycles = 1

    p = ps.Process(geometry, model, 1.0)
    p.setParameters(alpParams)
    p.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE)
    p.apply()

    # flux = p.calculateFlux()

    # points = np.array(flux.getNodes())
    # fluxValues = np.array(flux.getCellData().getScalarData("particleFlux", True))

    # cuts = points[:, 1] < 0.25
    # points = points[cuts]
    # fluxValues = fluxValues[cuts]

    # ps.ls.VTKWriter(flux, filename).apply()

    mesh = ps.ls.Mesh()
    ps.ToDiskMesh(geometry, mesh).apply()
    points = np.array(mesh.getNodes())
    cuts = points[:, 1] < 0.25
    points = points[cuts]
    return points
    # return points, fluxValues


p = run_simulation(0.1, "flux_0p1")
plt.plot(p[:, 0], p[:, 1], "--")
p = run_simulation(0.01, "flux_0p01")
plt.plot(p[:, 0], p[:, 1], ".-")
p = run_simulation(0.05, "flux_0p05")
plt.plot(p[:, 0], p[:, 1], "-")
plt.show()


# rayParams = ps.RayTracingParameters()
# rayParams.raysPerPoint = int(params["numRaysPerPoint"])

# ALP = ps.Process(geometry, model)
# ALP.setParameters(rayParams)
# ALP.setParameters(alpParams)
# ALP.apply()

# ## TODO: Implement MeasureProfile in Python
# #   MeasureProfile<NumericType, D>(domain, params.get("gapHeight") / 2.)
# #       .save(params.get<std::string>("outputFile"));

# geometry.saveVolumeMesh("SingleParticleALD_final")
