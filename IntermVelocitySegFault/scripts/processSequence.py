import viennaps as vps
import viennals as vls
import os
from types import SimpleNamespace


def processSequence(
    passedDomain: vps.Domain,
    params: dict[str, float],
    options=None,
):
    """
    Process a sequence of operations on the  passedDomain.

    Args:
        passedDomain (vls.passedDomain): The  passedDomain to process.

    """
    # Set default options if not provided
    if options is None:
        options = SimpleNamespace(
            rpp=500,
            fluxEngineType=vps.FluxEngineType.GPU_TRIANGLE,
            smoothingNeighbors=2,
            timeStepRatio=0.25,
            spatialScheme=vps.SpatialScheme.ENGQUIST_OSHER_1ST_ORDER,
            temporalScheme=vps.TemporalScheme.FORWARD_EULER,
            adaptiveTimeStepping=True,
            intermediateVelocityCalculations=False,
        )

    vps.Time.setUnit("s")  # Set time unit to seconds
    vps.Length.setUnit("nm")  # Set length unit to nanometers
    # Set up model parameters
    modelParams = vps.SF6C4F8Etching.defaultParameters()

    # Map parameters directly to model parameters
    modelParams.ionFlux = params["ionFlux"]
    modelParams.etchantFlux = params["etchantFlux"]
    modelParams.Ions.meanEnergy = params["meanEnergy"]
    # modelParams.Ions.sigmaEnergy = params["sigmaEnergy"]
    # modelParams.Ions.n_l = params["ionNl"]
    # modelParams.Ions.inflectAngle = params["ionInflectAngle"]
    modelParams.Substrate.A_ie = params["siAie"]
    # modelParams.Passivation.A_ie = params["passAie"]
    modelParams.Ions.exponent = params["ionExponent"]
    modelParams.passivationFlux = 0.0  # Set passivation flux to zero
    modelParams.Substrate.k_sigma = params["kSigma"]
    # modelParams.Substrate.k_sigma = 30.0
    modelParams.Polymer.B_sp = 0.0

    model = vps.SF6C4F8Etching(modelParams)
    model.setProcessName("PT-Step")
    process = vps.Process()

    # Configure and run the process
    process.setDomain(passedDomain)
    process.setProcessModel(model)
    process.setProcessDuration(0.33)

    rayTraceEtchStep = False

    rayTracing = vps.RayTracingParameters()
    rayTracing.raysPerPoint = options.rpp
    # rayTracing.useRandomSeeds = False
    # rayTracing.rngSeed = options.rngSeed
    rayTracing.smoothingNeighbors = options.smoothingNeighbors

    advection = vps.AdvectionParameters()
    advection.timeStepRatio = options.timeStepRatio
    # advection.adaptiveTimeStepping = options.adaptiveTimeStepping
    advection.spatialScheme = options.spatialScheme
    advection.temporalScheme = options.temporalScheme
    advection.calculateIntermediateVelocities = options.intermediateVelocityCalculations

    covParams = vps.CoverageParameters()
    covParams.tolerance = 1e-4

    process.setParameters(rayTracing)
    process.setParameters(advection)
    process.setParameters(covParams)
    process.setFluxEngineType(options.fluxEngineType)

    if rayTraceEtchStep:
        modelParamsEtchProper = modelParams
        modelParamsEtchProper.ionFlux = 0.0
        modelParamsEtchProper.etchantFlux = params["etchantFluxEtchStep"]
        modelEtchProper = vps.SF6C4F8Etching(modelParamsEtchProper)
        modelEtchProper.setProcessName("Etch-Step-Proper")
        processEtchProper = vps.Process()
        processEtchProper.setDomain(passedDomain)
        processEtchProper.setProcessModel(modelEtchProper)
        processEtchProper.setProcessDuration(0.19)

        rayTracing2 = vps.RayTracingParameters()
        rayTracing2.raysPerPoint = options.rpp

        processEtchProper.setParameters(rayTracing2)
        processEtchProper.setFluxEngineType(vps.FluxEngineType.GPU_TRIANGLE)

    else:
        directionDown = [0.0, -1.0, 0.0]

        # Define directional rate
        etchDirDown = vps.RateSet(
            calculateVisibility=False,
            direction=directionDown,
            directionalVelocity=0.0,
            isotropicVelocity=-params["isoEtchDepth"],
            maskMaterials=[vps.Material.Mask, vps.Material.Polymer],
        )
        etchProcess = vps.DirectionalProcess(rateSets=[etchDirDown])

    isoDep = vps.IsotropicProcess(1.0)
    isoDep.setProcessName("IsotropicDeposition")

    saveSubsteps = True
    if saveSubsteps:
        substepsDebugFolder = (
            os.path.dirname(os.path.abspath(__file__)) + "/substepsDebug/"
        )
        os.makedirs(substepsDebugFolder, exist_ok=True)

    numCycles = 1
    passedDomain.duplicateTopLevelSet(vps.Material.Polymer)
    for j in range(numCycles):
        vps.Process(passedDomain, isoDep, params["depThickness"]).apply()
        if saveSubsteps:
            passedDomain.saveSurfaceMesh(
                (f"{substepsDebugFolder}substep-remove30_{j + 1}_dep.vtp"),
                True,
            )

        process.apply()
        if saveSubsteps:
            passedDomain.saveSurfaceMesh(
                (f"{substepsDebugFolder}substep-remove30_{j + 1}_pt.vtp"),
                True,
            )

        if rayTraceEtchStep:
            processEtchProper.apply()
        else:
            vps.Process(passedDomain, etchProcess, 1).apply()

        # processEtchProper.apply()
        if saveSubsteps:
            passedDomain.saveSurfaceMesh(
                (f"{substepsDebugFolder}substep-remove30_{j + 1}_etch.vtp"),
                True,
            )

    passedDomain.removeTopLevelSet()

    vps.Planarize(passedDomain, 0.0).apply()

    resultLS = passedDomain.getLevelSets()[-1]

    return resultLS, passedDomain
