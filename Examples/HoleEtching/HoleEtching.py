# switch between 2D and 3D mode
DIM = 3

if DIM == 2:
    import viennaps2d as vps
else:
    import viennaps3d as vps

# print intermediate output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)

# parse parameters
params = vps.psReadConfigFile("config.txt")

# geometry setup
geometry = vps.Domain()
vps.MakeHole(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    holeRadius=params["holeRadius"],
    holeDepth=params["maskHeight"],
    taperingAngle=params["taperAngle"],
    makeMask=True,
    material=vps.Material.Si,
).apply()

# use pre-defined model SF6O2 etching model
model = vps.SF6O2Etching(
    ionFlux=params["ionFlux"],
    etchantFlux=params["etchantFlux"],
    oxygenFlux=params["oxygenFlux"],
    meanIonEnergy=params["meanEnergy"],
    sigmaIonEnergy=params["sigmaEnergy"],
    oxySputterYield=params["A_O"],
    etchStopDepth=params["etchStopDepth"],
)

# process setup
process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setMaxCoverageInitIterations(10)
process.setNumberOfRaysPerPoint(int(params["raysPerPoint"]))
process.setProcessDuration(params["processTime"])

# print initial surface
geometry.printSurface(filename="initial.vtp", addMaterialIds=True)

# run the process
process.apply()

# print final surface
geometry.printSurface(filename="final.vtp", addMaterialIds=True)
