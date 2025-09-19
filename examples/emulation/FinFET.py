import viennaps as ps

ps.setDimension(3)
volumeOutput = False
volumeNum = 0
surfaceNum = 0


def writeVolume(domain):
    if not volumeOutput:
        return
    global volumeNum
    print("Writing volume mesh ...", end="", flush=True)
    domain.saveVolumeMesh("FinFET_" + str(volumeNum))
    print(" done")
    volumeNum += 1


def writeSurface(domain):
    global surfaceNum
    domain.saveSurfaceMesh("FinFET_" + str(surfaceNum) + ".vtp", True)
    surfaceNum += 1


ps.Logger.setLogLevel(ps.LogLevel.ERROR)

boundaryConds = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]
bounds = [0.0, 90.0, 0.0, 100.0, 0.0, 70.0]  # in nanometers
gridDelta = 0.79

domain = ps.Domain(bounds, boundaryConds, gridDelta)

# Initialise domain with a single silicon plane (at z=70 because it is 70 nm high)
ps.MakePlane(domain, 70.0, ps.Material.Si).apply()
writeSurface(domain)


# Add double patterning mask
box = ps.ls.Domain(domain.getGrid())
minPoints = [30.0, -10.0, 69.9]
maxPoints = [60.0, 110.0, 90.0]
geo = ps.ls.MakeGeometry(box, ps.ls.Box(minPoints, maxPoints)).apply()
domain.insertNextLevelSetAsMaterial(box, ps.Material.Mask)
writeSurface(domain)

#  Double patterning processes
print("Double patterning processes ...", end="", flush=True)
thickness = 4.0  # nm
domain.duplicateTopLevelSet(ps.Material.Metal)
growth = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, growth, 0).apply()
print(" done")
writeSurface(domain)

# DP-Patterning
print("DP-Patterning ...", end="", flush=True)
etchDepth = 6.0  # nm
dist = ps.BoxDistribution(
    [-gridDelta, -gridDelta, -etchDepth], gridDelta, domain.getLevelSets()[0]
)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

# Remove mask with boolean operation
domain.removeMaterial(ps.Material.Mask)
writeSurface(domain)
writeVolume(domain)

# pattern si
print("Si-Patterning ...", end="", flush=True)
etchDepth = 90.0  # nm
direction = [0.0, 0.0, 1.0]
model = ps.DirectionalProcess(
    direction=direction,
    directionalVelocity=1.1,
    isotropicVelocity=0.1,
    maskMaterial=ps.Material.Metal,
    calculateVisibility=False,
)
ps.Process(domain, model, etchDepth).apply()
print(" done")
writeVolume(domain)
writeSurface(domain)

# Remove DP mask (metal)
domain.removeTopLevelSet()
writeSurface(domain)

# deposit STI material
print("STI Deposition ...", end="", flush=True)
thickness = 35  # nm
domain.duplicateTopLevelSet(ps.Material.SiO2)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

# CMP at 80
ps.Planarize(domain, 80.0).apply()
writeVolume(domain)

# pattern STI material
print("STI Patterning ...", end="", flush=True)
dist = ps.SphereDistribution(
    radius=-35, gridDelta=gridDelta, mask=domain.getLevelSets()[0]
)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)
writeVolume(domain)

# deposit gate material
print("Gate Deposition HfO2 ...", end="", flush=True)
thickness = 2.0  # nm
domain.duplicateTopLevelSet(ps.Material.HfO2)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

print("Gate Deposition PolySi ...", end="", flush=True)
thickness = 104.0  # nm
domain.duplicateTopLevelSet(ps.Material.PolySi)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

# CMP at 150
ps.Planarize(domain, 150.0).apply()

# dummy gate mask addition
box = ps.ls.Domain(domain.getGrid())
minPoint = [-10.0, 30.0, 145.0]
maxPoint = [100.0, 70.0, 175.0]
geo = ps.ls.MakeGeometry(box, ps.ls.Box(minPoint, maxPoint)).apply()
domain.insertNextLevelSetAsMaterial(box, ps.Material.Mask)
writeSurface(domain)

# gate patterning
print("Dummy Gate Patterning ...", end="", flush=True)
direction = [0.0, 0.0, 1.0]
masks = [ps.Material.Mask, ps.Material.Si, ps.Material.SiO2]
model = ps.DirectionalProcess(
    direction=direction,
    directionalVelocity=1.0,
    isotropicVelocity=0.0,
    maskMaterial=masks,
    calculateVisibility=False,
)
ps.Process(domain, model, 110.0).apply()
print(" done")
writeSurface(domain)

# Remove mask
domain.removeTopLevelSet()
writeSurface(domain)
writeVolume(domain)

# Spacer Deposition and Etch
print("Spacer Deposition and Etch ...", end="", flush=True)
thickness = 10.0  # nm
domain.duplicateTopLevelSet(ps.Material.Si3N4)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

# Spacer Etch
print("Spacer Etch ...", end="", flush=True)
ls = domain.getLevelSets()[-2]
dist = ps.BoxDistribution(
    halfAxes=[-gridDelta, -gridDelta, -50], gridDelta=gridDelta, mask=ls
)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)
writeVolume(domain)

# isotropic etch (fin-release)
print("Fin-Release ...", end="", flush=True)
masks = [ps.Material.PolySi, ps.Material.SiO2, ps.Material.Si3N4]
model = ps.IsotropicProcess(rate=-1.0, maskMaterial=masks)
advParams = ps.AdvectionParameters()
advParams.integrationScheme = ps.IntegrationScheme.LAX_FRIEDRICHS_2ND_ORDER
process = ps.Process(domain, model, 5.0)
process.setParameters(advParams)
process.apply()
print(" done")
writeSurface(domain)
writeVolume(domain)

# source/drain epitaxy
print("S/D Epitaxy ...", end="", flush=True)
domain.duplicateTopLevelSet(ps.Material.SiGe)
advectionParams = ps.AdvectionParameters()
advectionParams.integrationScheme = (
    ps.IntegrationScheme.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)
ps.StencilLocalLaxFriedrichsScalar.setMaxDissipation(1000)
material = [
    (ps.Material.Si, 1.0),
    (ps.Material.SiGe, 1.0),
]
model = ps.SelectiveEpitaxy(materialRates=material)
process = ps.Process(domain, model, 14.0)
process.setParameters(advectionParams)
process.apply()
print(" done")
writeSurface(domain)
writeVolume(domain)

# deposit dielectric
print("Dielectric Deposition ...", end="", flush=True)
thickness = 50.0  # nm
domain.duplicateTopLevelSet(ps.Material.Dielectric)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

# CMP at 90
ps.Planarize(domain, 90.0).apply()
writeVolume(domain)

# now remove gate and add new gate materials
domain.removeMaterial(ps.Material.PolySi)
writeSurface(domain)
writeVolume(domain)

# now deposit TiN and PolySi as replacement gate
print("Gate Deposition TiN ...", end="", flush=True)
thickness = 4.0  # nm
domain.duplicateTopLevelSet(ps.Material.TiN)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")
writeSurface(domain)

print("Gate Deposition PolySi ...", end="", flush=True)
thickness = 40.0  # nm
domain.duplicateTopLevelSet(ps.Material.PolySi)
dist = ps.SphereDistribution(radius=thickness, gridDelta=gridDelta)
ps.Process(domain, dist, 0).apply()
print(" done")

# CMP at 90
ps.Planarize(domain, 90.0).apply()
writeVolume(domain)

domain.saveVolumeMesh("FinFET_Final", 0.05)
