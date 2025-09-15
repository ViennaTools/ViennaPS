import numpy as np
import viennaps as ps

ps.setNumThreads(16)
ps.setDimension(3)
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit("um")
ps.Time.setUnit("min")

ps.gpu.Context.createContext()

# hole geometry parameters
gridDelta = 0.03  # um
extent = 1.5
holeRadius = 0.175
maskHeight = 1.2
taperAngle = 1.193

# fluxes
ionFlux = [10.0, 10.0, 10.0, 10.0, 10.0]
etchantFlux = [4.8e3, 4.5e3, 4e3, 3.5e3, 4e3]
oxygenFlux = [3e2, 8e2, 2e3, 2.5e3, 0.0]
A_O = [2, 2, 2, 1, 1]
yo2 = [0.44, 0.5, 0.56, 0.62, 0]

# etching model parameters
params = ps.SF6O2Etching.defaultParameters()
params.Substrate.A_ie = 5.0
params.Substrate.Eth_ie = 15.0

params.Substrate.A_sp = 0.0337
params.Substrate.Eth_sp = 20.0

params.Ions.exponent = 500
params.Ions.meanEnergy = 100.0
params.Ions.sigmaEnergy = 10.0
params.Ions.minAngle = np.deg2rad(85.0)
params.Ions.inflectAngle = np.deg2rad(89.0)

params.Mask.rho = params.Substrate.rho * 10.0

# simulation parameters
processDuration = 3  # min
integrationScheme = ps.IntegrationScheme.ENGQUIST_OSHER_2ND_ORDER
numberOfRaysPerPoint = int(1000)

for i in range(len(yo2)):

    # geometry setup, all units in um
    geometry = ps.Domain(
        gridDelta=gridDelta,
        xExtent=extent,
        yExtent=extent,
    )
    ps.MakeHole(
        domain=geometry,
        holeRadius=holeRadius,
        holeDepth=0.0,
        maskHeight=maskHeight,
        maskTaperAngle=taperAngle,
        holeShape=ps.HoleShape.HALF,
    ).apply()

    rayParams = ps.RayTracingParameters()
    rayParams.smoothingNeighbors = 2
    rayParams.raysPerPoint = numberOfRaysPerPoint

    advectionParams = ps.AdvectionParameters()
    advectionParams.integrationScheme = integrationScheme

    coverageParams = ps.CoverageParameters()
    coverageParams.maxIterations = 20
    coverageParams.coverageDeltaThreshold = 1e-4

    process = ps.Process()
    process.setDomain(geometry)
    process.setProcessDuration(processDuration)
    process.setRayTracingParameters(rayParams)
    process.setAdvectionParameters(advectionParams)
    process.setCoverageParameters(coverageParams)
    process.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE)

    params.ionFlux = ionFlux[i]
    params.etchantFlux = etchantFlux[i]
    params.passivationFlux = oxygenFlux[i]
    params.Passivation.A_ie = A_O[i]

    model = ps.SF6O2Etching(params)

    process.setProcessModel(model)
    process.apply()

    geometry.saveSurfaceMesh("hole_y{:.2f}.vtp".format(yo2[i]), True)
