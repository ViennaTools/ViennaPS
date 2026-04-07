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
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
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
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
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
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Deposit dummy gate material
print("Depositing dummy gate material ...")
domain.duplicateTopLevelSet(ps.Material.PolySi)
ps.Process(domain, growth, 85.0).apply()

# CMP at 100nm height
ps.Planarize(domain, 100.0).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Dummy gate mask addition
print("Adding dummy gate mask ...")
mask = ps.ls.Domain(bounds, boundaryConds, gridDelta)
geo = ps.ls.MakeGeometry(mask, ps.ls.Box([-10, 30, 99], [80, 70, 105]))
geo.setIgnoreBoundaryConditions(True)
geo.apply()
domain.insertNextLevelSetAsMaterial(mask, ps.Material.Mask)
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

masks = domain.getMaterialsInDomain()
masks.remove(ps.Material.PolySi)
geometricEtch = ps.BoxDistribution([-gridDelta, -gridDelta, -100])
for material in masks:
    geometricEtch.addMaskMaterial(material)

ps.Process(domain, geometricEtch).apply()

domain.removeMaterial(ps.Material.Mask)
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Spacer Deposition
print("Depositing spacer material ...")
domain.duplicateTopLevelSet(ps.Material.Si3N4)
ps.Process(domain, growth, 10.0).apply()
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Spacer patterning
print("Patterning spacer ...")
directional = ps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0], materialRates={ps.Material.Si3N4: [1.0, 0.0]}
)
ps.Process(domain, directional, 40.0).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Fin patterning
print("Patterning Fin ...")
directional = ps.DirectionalProcess(
    direction=[0.0, 0.0, -1.0],
    materialRates={
        ps.Material.PolySi: [0.1, 0.0],
        ps.Material.Si: [1.0, 0.0],
        ps.Material.SiGe: [1.0, 0.0],
    },
    defaultIsotropicRate=-0.005,
)
ps.Process(domain, directional, 21.0).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# SD Epitaxy
print("SD Epitaxy ...")
epiMaterial = ps.MaterialMap.fromString("EpiSi")
domain.duplicateTopLevelSet(epiMaterial)
epitaxy = ps.IsotropicProcess(0.0)
epitaxy.setMaterialRate(ps.Material.Si, 1.0)
epitaxy.setMaterialRate(epiMaterial, 1.0)
ps.Process(domain, epitaxy, 9.0).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Dielectric deposition
print("Depositing dielectric ...")
domain.duplicateTopLevelSet(ps.Material.Dielectric)
ps.Process(domain, growth, 35.0).apply()

ps.Planarize(domain, 60).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# Remove dummy gate
print("Removing dummy gate ...")
domain.removeMaterial(ps.Material.PolySi)
domain.saveSurfaceMesh("StackedNanowire_" + str(n))
n += 1

# remove SiGe interlayer
print("Removing SiGe interlayer ...")
etch = ps.IsotropicProcess(rate=0.0)  # default rate is 0.0
etch.setMaterialRate(ps.Material.SiGe, -1.0)
ps.Process(domain, etch, 10.0).apply()

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
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

domain.saveSurfaceMesh("StackedNanowire_" + str(n))
domain.saveVolumeMesh("StackedNanowire_Final")
