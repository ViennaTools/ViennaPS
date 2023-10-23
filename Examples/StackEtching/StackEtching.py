import viennaps2d as vps

# Set process verbosity
vps.psLogger.setLogLevel(vps.psLogLevel.INTERMEDIATE)

# Parse process parameters
params = vps.psReadConfigFile("config.txt")

# Geometry setup
geometry = vps.psDomain()
vps.psMakeStack(geometry, gridDelta=params['gridDelta'], xExtent=params['xExtent'], yExtent=params['yExtent'], 
                numLayers=int(params['numLayers']), layerHeight=params['layerHeight'],
                substrateHeight=params['substrateHeight'], holeRadius=params['holeRadius'], 
                maskHeight=params['maskHeight'], periodicBoundary=False).apply()

geometry.duplicateTopLevelSet(vps.psMaterial.Polymer)

model = vps.FluorocarbonEtching(ionFlux=params['ionFlux'], etchantFlux=params['etchantFlux'], polyFlux=params['polyFlux'],
                                meanIonEnergy=params['meanIonEnergy'], sigmaIonEnergy=params['sigmaIonEnergy'],
                                ionExponent=params['ionExponent'],)

process = vps.psProcess()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params['processTime'])
process.setMaxCoverageInitIterations(10)
process.setTimeStepRatio(0.25)

# print initial surface
vps.psWriteVisualizationMesh(geometry, "initial").apply()

process.apply()

# print final surface
vps.psWriteVisualizationMesh(geometry, "final").apply()