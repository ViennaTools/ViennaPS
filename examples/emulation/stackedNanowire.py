import viennaps as ps

ps.setDimension(3)
ps.Logger.setLogLevel(ps.LogLevel.ERROR)
n = 0

# domain
bounds = [0.0, 70.0, 0.0, 100.0, 0.0, 70.0]
boundaryConds = [
    ps.BoundaryType.PERIODIC_BOUNDARY,
    ps.BoundaryType.PERIODIC_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]
gridDelta = 0.71

domain = ps.Domain(bounds, boundaryConds, gridDelta)
ps.MakePlane(domain, 10.0, ps.Material.SiO2).apply()

# Epi Fin growth
print("Epi Fin growth ...")
growth = ps.IsotropicProcess(1.0)
domain.duplicateTopLevelSet(ps.Material.Si)
ps.Process(domain, growth, 7.0).apply()

domain.duplicateTopLevelSet(ps.Material.SiGe)
ps.Process(domain, growth, 8.0).apply()

domain.duplicateTopLevelSet(ps.Material.Si)
ps.Process(domain, growth, 7.0).apply()

# Add double patterning mask
print("Adding double patterning mask ...")
mask = ps.ls.Domain(bounds, boundaryConds, gridDelta)
geo = ps.ls.MakeGeometry(mask, ps.ls.Box([25.0, -10.0, 31.9], [45.0, 110.0, 60.0]))
geo.setIgnoreBoundaryConditions(True)
geo.apply()

domain.insertNextLevelSetAsMaterial(mask, ps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Double patterning
domain.duplicateTopLevelSet(ps.Material.Metal)
ps.Process(domain, growth, 15.0).apply()

directional = ps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0],
    directionalVelocity=-1.0,
    isotropicVelocity=0.0,
    maskMaterial=[ps.Material.Si, ps.Material.Mask],
    calculateVisibility=False,
)
ps.Process(domain, directional, 20.0).apply()

domain.removeMaterial(ps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Pattern Si/SiGe/Si
print("Patterning Si/SiGe/Si Stack ...")
directional = ps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0],
    directionalVelocity=-1.0,
    isotropicVelocity=0.0,
    maskMaterial=ps.Material.Metal,
    calculateVisibility=False,
)
ps.Process(domain, directional, 30.0).apply()

domain.removeMaterial(ps.Material.Metal)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Deposit dummy gate material
print("Depositing dummy gate material ...")
domain.duplicateTopLevelSet(ps.Material.PolySi)
ps.Process(domain, growth, 55.0).apply()

# CMP at 80nm height
ps.Planarize(domain, 80.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Dummy gate mask addition
print("Adding dummy gate mask ...")
mask = ps.ls.Domain(bounds, boundaryConds, gridDelta)
geo = ps.ls.MakeGeometry(mask, ps.ls.Box([-10, 30, 75], [80, 70, 80]))
geo.setIgnoreBoundaryConditions(True)
geo.apply()

ps.ls.BooleanOperation(
    mask, domain.getLevelSets()[-2], ps.ls.BooleanOperationEnum.UNION
).apply()

tmpDomain = ps.Domain()
tmpDomain.insertNextLevelSetAsMaterial(mask, ps.Material.Mask)
tmpDomain.insertNextLevelSetAsMaterial(domain.getLevelSets()[-1], ps.Material.PolySi)

geometricEtch = ps.BoxDistribution([-gridDelta, -gridDelta, -80], gridDelta, mask)
ps.Process(tmpDomain, geometricEtch, 1.0).apply()

domain.removeMaterial(ps.Material.Mask)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Spacer Deposition
print("Depositing spacer material ...")
domain.duplicateTopLevelSet(ps.Material.Si3N4)
ps.Process(domain, growth, 12.0).apply()

# Spacer patterning
print("Patterning spacer ...")
directional = ps.DirectionalProcess(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=1.0,
    isotropicVelocity=-0.05,
    maskMaterial=[
        ps.Material.PolySi,
        ps.Material.Si,
        ps.Material.SiGe,
        ps.Material.SiO2,
    ],
    calculateVisibility=False,
)
ps.Process(domain, directional, 40.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Fin patterning
print("Patterning Fin ...")
rateSet1 = ps.RateSet(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=1.0,
    isotropicVelocity=0.0,
    maskMaterials=[ps.Material.PolySi, ps.Material.SiO2, ps.Material.Si3N4],
    calculateVisibility=False,
)
rateSet2 = ps.RateSet(
    direction=[0.0, 0.0, 1.0],
    directionalVelocity=0.1,
    isotropicVelocity=0.0,
    maskMaterials=[
        ps.Material.SiO2,
        ps.Material.Si3N4,
        ps.Material.Si,
        ps.Material.SiGe,
    ],
    calculateVisibility=False,
)
directional = ps.DirectionalProcess([rateSet1, rateSet2])
ps.Process(domain, directional, 21.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# SD Epitaxy
print("SD Epitaxy ...")
domain.duplicateTopLevelSet(ps.Material.Metal)
epitaxy = ps.IsotropicProcess(
    rate=1.0,
    maskMaterial=[ps.Material.PolySi, ps.Material.SiO2, ps.Material.Si3N4],
)
ps.Process(domain, epitaxy, 9.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Dielectric deposition
print("Depositing dielectric ...")
domain.duplicateTopLevelSet(ps.Material.Dielectric)
ps.Process(domain, growth, 35.0).apply()

ps.Planarize(domain, 60).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# Remove dummy gate
print("Removing dummy gate ...")
domain.removeMaterial(ps.Material.PolySi)
domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# remove SiGe interlayer
print("Removing SiGe interlayer ...")
etch = ps.IsotropicProcess(
    rate=-1.0,
    maskMaterial=[
        ps.Material.Si,
        ps.Material.SiO2,
        ps.Material.Si3N4,
        ps.Material.Dielectric,
        ps.Material.Metal,
    ],
)
ps.Process(domain, etch, 10.0).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
n += 1

# gate dielectric, gate metal, gate contact deposition
print("Depositing gate materials ...")
domain.duplicateTopLevelSet(ps.Material.HfO2)
ps.Process(domain, growth, 2.0).apply()

domain.duplicateTopLevelSet(ps.Material.TiN)
ps.Process(domain, growth, 4.0).apply()

domain.duplicateTopLevelSet(ps.Material.PolySi)
ps.Process(domain, growth, 20.0).apply()

ps.Planarize(domain, 47.5).apply()

domain.saveVolumeMesh("StackedNanowire_" + str(n))
