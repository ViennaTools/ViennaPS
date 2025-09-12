import viennaps.d3 as psd
import viennaps as ps

ps.Logger.setLogLevel(ps.LogLevel.INFO)

context = ps.gpu.Context.createContext()

# Create a trench
gridDelta = 1.0
extent = 50.0
trenchWidth = 15.0
maskHeight = 40.0

domain = psd.Domain()
psd.MakeTrench(
    domain=domain,
    gridDelta=gridDelta,
    xExtent=extent,
    yExtent=extent,
    trenchWidth=trenchWidth,
    trenchDepth=maskHeight,
    baseHeight=0.0,
    taperingAngle=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=ps.Material.Si,
).apply()
domain.saveSurfaceMesh("trench_initial.vtp")

time = 30.0
sticking = 0.1
rate = 1.0
exponent = 1.0

# Create a trench depo model
model = psd.SingleParticleProcess(rate, sticking, exponent)

rtParams = ps.RayTracingParameters()
rtParams.smoothingNeighbors = 2
rtParams.raysPerPoint = 5000

process = psd.Process()
process.setDomain(domain)
process.setProcessModel(model)
process.setProcessDuration(time)
process.setRayTracingParameters(rtParams)
process.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE)

process.apply()

domain.saveSurfaceMesh("trench_final.vtp")
