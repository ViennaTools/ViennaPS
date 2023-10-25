# switch between 2D and 3D mode
DIM = 2

if DIM == 2:
    import viennaps2d as vps
else:
    import viennaps3d as vps

params = vps.psReadConfigFile("config.txt")

geometry = vps.psDomain()
vps.psMakeTrench(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    taperingAngle=params["taperAngle"],
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=False,
    material=vps.psMaterial.Si,
).apply()

geometry.duplicateTopLevelSet(vps.psMaterial.SiO2)

model = vps.SimpleDeposition(
    stickingProbability=params["stickingProbability"],
    sourceExponent=params["sourcePower"],
)

process = vps.psProcess()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])

geometry.printSurface("initial.vtp")

process.apply()

geometry.printSurface("final.vtp")

if DIM == 2:
    vps.psWriteVisualizationMesh(geometry, "final").apply()
