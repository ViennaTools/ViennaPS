import viennaps3d as vps
import viennals3d as vls

vps.Logger.setLogLevel(vps.LogLevel.ERROR)
n = 0

# domain
bounds = [0.0, 70.0, 0.0, 100.0, 0.0, 70.0]
boundaryConds = [
    vls.BoundaryConditionEnum.PERIODIC_BOUNDARY,
    vls.BoundaryConditionEnum.PERIODIC_BOUNDARY,
    vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]
gridDelta = 0.71

domain = vps.Domain(bounds, boundaryConds, gridDelta)
vps.MakePlane(domain, 10.0, vps.Material.SiO2).apply()

# Epi Fin growth
print("Epi Fin growth ...")
growth = vps.IsotropicProcess(1.0)
domain.duplicateTopLevelSet(vps.Material.Si)
vps.Process(domain, growth, 7.0).apply()

domain.duplicateTopLevelSet(vps.Material.SiGe)
vps.Process(domain, growth, 8.0).apply()

domain.duplicateTopLevelSet(vps.Material.Si)
vps.Process(domain, growth, 7.0).apply()

# Add double patterning mask
print("Adding double patterning mask ...")
mask = vls.Domain(bounds, boundaryConds, gridDelta)
geo = vls.MakeGeometry(mask, vls.Box([25.0, -10.0, 31.9], [45.0, 110.0, 60.0]))
geo.setIgnoreBoundaryConditions(True)
geo.apply()

domain.insertNextLevelSetAsMaterial(mask, vps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Double patterning
domain.duplicateTopLevelSet(vps.Material.Metal)
vps.Process(domain, growth, 15.0).apply()

directional = vps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0],
    directionalVelocity=-1.0,
    isotropicVelocity=0.0,
    maskMaterial=[vps.Material.Si, vps.Material.Mask],
    calculateVisibility=False,
)
vps.Process(domain, directional, 20.0).apply()

domain.removeMaterial(vps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Pattern Si/SiGe/Si
print("Patterning Si/SiGe/Si Stack ...")
directional = vps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0],
    directionalVelocity=-1.0,
    isotropicVelocity=0.0,
    maskMaterial=vps.Material.Metal,
    calculateVisibility=False,
)
vps.Process(domain, directional, 30.0).apply()

domain.removeMaterial(vps.Material.Metal)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Deposit dummy gate material
print("Depositing dummy gate material ...")
domain.duplicateTopLevelSet(vps.Material.PolySi)
vps.Process(domain, growth, 55.0).apply()

# CMP at 80nm height
vps.Planarize(domain, 80.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Dummy gate mask addition
print("Adding dummy gate mask ...")
mask = vls.Domain(bounds, boundaryConds, gridDelta)
geo = vls.MakeGeometry(mask, vls.Box([-10, 30, 75], [80, 70, 80]))
geo.setIgnoreBoundaryConditions(True)
geo.apply()

vls.BooleanOperation(
    mask, domain.getLevelSets()[-2], vls.BooleanOperationEnum.UNION
).apply()

tmpDomain = vps.Domain()
tmpDomain.insertNextLevelSetAsMaterial(mask, vps.Material.Mask)
tmpDomain.insertNextLevelSetAsMaterial(domain.getLevelSets()[-1], vps.Material.PolySi)

geometricEtch = vps.BoxDistribution([-gridDelta, -gridDelta, -80], gridDelta, mask)
vps.Process(tmpDomain, geometricEtch, 1.0).apply()

domain.removeMaterial(vps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Spacer Deposition
print("Depositing spacer material ...")
domain.duplicateTopLevelSet(vps.Material.Si3N4)
vps.Process(domain, growth, 12.0).apply()

# Spacer patterning
print("Patterning spacer ...")
directional = vps.DirectionalProcess(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=1.0,
    isotropicVelocity=-0.05,
    maskMaterial=[
        vps.Material.PolySi,
        vps.Material.Si,
        vps.Material.SiGe,
        vps.Material.SiO2,
    ],
    calculateVisibility=False,
)
vps.Process(domain, directional, 40.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Fin patterning
print("Patterning Fin ...")
rateSet1 = vps.RateSet(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=1.0,
    isotropicVelocity=0.0,
    maskMaterials=[vps.Material.PolySi, vps.Material.SiO2, vps.Material.Si3N4],
    calculateVisibility=False,
)
rateSet2 = vps.RateSet(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=0.1,
    isotropicVelocity=0.0,
    maskMaterials=[
        vps.Material.SiO2,
        vps.Material.Si3N4,
        vps.Material.Si,
        vps.Material.SiGe,
    ],
    calculateVisibility=False,
)
directional = vps.DirectionalProcess([rateSet1, rateSet2])
vps.Process(domain, directional, 21.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# SD Epitaxy
print("SD Epitaxy ...")
domain.duplicateTopLevelSet(vps.Material.Metal)
epitaxy = vps.IsotropicProcess(
    rate=1.0,
    maskMaterial=[vps.Material.PolySi, vps.Material.SiO2, vps.Material.Si3N4],
)
vps.Process(domain, epitaxy, 9.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Dielectric deposition
print("Depositing dielectric ...")
domain.duplicateTopLevelSet(vps.Material.Dielectric)
vps.Process(domain, growth, 35.0).apply()

vps.Planarize(domain, 60).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Remove dummy gate
print("Removing dummy gate ...")
domain.removeMaterial(vps.Material.PolySi)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# remove SiGe interlayer
print("Removing SiGe interlayer ...")
etch = vps.IsotropicProcess(
    rate=-1.0,
    maskMaterial=[
        vps.Material.Si,
        vps.Material.SiO2,
        vps.Material.Si3N4,
        vps.Material.Dielectric,
        vps.Material.Metal,
    ],
)
vps.Process(domain, etch, 10.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# gate dielectric, gate metal, gate contact deposition
print("Depositing gate materials ...")
domain.duplicateTopLevelSet(vps.Material.HfO2)
vps.Process(domain, growth, 2.0).apply()

domain.duplicateTopLevelSet(vps.Material.TiN)
vps.Process(domain, growth, 4.0).apply()

domain.duplicateTopLevelSet(vps.Material.PolySi)
vps.Process(domain, growth, 20.0).apply()

vps.Planarize(domain, 47.5).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
