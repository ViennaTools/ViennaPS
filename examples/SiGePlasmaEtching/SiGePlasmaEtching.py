import viennaps2d as vps
import viennals2d as vls

from Parameters import Parameters, read_config_file
import Geometry
import ReadPath

# ----------------------
# Load configuration
# ----------------------
import sys

vps.Logger.setLogLevel(vps.LogLevel.ERROR)
vls.Logger.setLogLevel(vls.LogLevel.ERROR)

config = read_config_file(sys.argv[1])
params = Parameters.from_dict(config)
geometry = vps.Domain()
Geometry.make_initial_geometry(geometry, params)
geometry.saveVolumeMesh(params.fileName + "Initial")

target = vps.Domain(geometry)
target.deepCopy(geometry)

# ----------------------
# Import CSV
# ----------------------
print("Importing path from:", params.pathFile)
ReadPath.read_path_dual(geometry, target, params)

target.saveSurfaceMesh(params.fileName + "Target.vtp")
if params.saveVolume:
    geometry.saveVolumeMesh(params.fileName + "Start")
    target.saveVolumeMesh(params.fileName + "Target")

# ----------------------
# Plasma Etching with CF4/O2
# ----------------------
procParams = vps.ReadConfigFile(sys.argv[1])

vps.Length.setUnit(procParams["lengthUnit"])
vps.Time.setUnit(procParams["timeUnit"])

modelParams = vps.CF4O2Parameters()
modelParams.ionFlux=procParams["ionFlux"]
modelParams.etchantFlux=procParams["etchantFlux"]
modelParams.oxygenFlux=procParams["oxygenFlux"]
modelParams.polymerFlux=procParams["polymerFlux"]
modelParams.Ions.meanEnergy = procParams["meanEnergy"]
modelParams.Ions.sigmaEnergy = procParams["sigmaEnergy"]
modelParams.Passivation.A_O_ie = procParams["A_O"]
modelParams.Passivation.A_C_ie = procParams["A_C"]

model = vps.CF4O2Etching(modelParams)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setMaxCoverageInitIterations(20)
process.setCoverageDeltaThreshold(1e-4)
process.setNumberOfRaysPerPoint(int(procParams["numRaysPerPoint"]))
process.setProcessDuration(procParams["processTime"])
process.setIntegrationScheme(
    vps.util.convertIntegrationScheme(procParams["integrationScheme"])
)

print("Etching with CF4/O2 plasma ...")
geometry.saveSurfaceMesh("etched_0.vtp", True)

for i in range(1, int(procParams["numCycles"] + 1)):
    process.apply()
    geometry.saveSurfaceMesh(f"etched_{i}.vtp", True)

geometry.saveVolumeMesh(params.fileName + "Final")

# ----------------------
# Compare target vs result
# ----------------------
compare = vls.CompareSparseField(geometry.getLevelSets()[-1], target.getLevelSets()[-1])

targetls = vls.Domain(target.getLevelSets()[-1])
meshtarget = vls.Mesh()
vls.ToMesh(targetls, meshtarget).apply()
vls.VTKWriter(meshtarget, "target.vtp").apply()

geometryls = vls.Domain(geometry.getLevelSets()[-1])
meshgeom = vls.Mesh()
vls.ToMesh(geometryls, meshgeom).apply()
vls.VTKWriter(meshgeom, "geometry.vtp").apply()

comparisonMesh = vls.Mesh()
compare.setOutputMesh(comparisonMesh)

compare.apply()
vls.VTKWriter(comparisonMesh, "comparison.vtp").apply()
print("sumSqDiff:", compare.getSumSquaredDifferences())
print("sumDiff:", compare.getSumDifferences())
