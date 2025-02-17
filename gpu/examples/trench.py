import viennaps3d as vps

vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

context = vps.gpu.Context()
context.create()

# Create a trench
gridDelta = 1.0
extent = 50.0
trenchWidth = 15.0
maskHeight = 40.0

domain = vps.Domain()
vps.MakeTrench(
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
    material=vps.Material.Si,
).apply()
domain.saveSurfaceMesh("trench_initial.vtp")

time = 30.0
sticking = 0.1
rate = 1.0
exponent = 1.0

# Create a trench depo model
model = vps.gpu.SingleParticleProcess(
    rate=rate, stickingProbability=sticking, sourceExponent=exponent
)

process = vps.gpu.Process(context)
process.setDomain(domain)
process.setProcessModel(model)
process.setProcessDuration(time)

process.apply()

domain.saveSurfaceMesh("trench_final.vtp")
