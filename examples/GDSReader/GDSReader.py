import viennaps3d as vps

try:
    # ViennaLS Python bindings are needed for the extrusion tool
    import viennals3d as vls
except ModuleNotFoundError:
    print("ViennaLS Python module not found. Can not parse GDS file.")
    exit(1)

vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

gridDelta = 0.01
boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

mask = vps.GDSGeometry(gridDelta)
mask.setBoundaryConditions(boundaryConds)
mask.addBlur([5., 80.], [0.8, 0.2], 0.5, gridDelta)

reader = vps.GDSReader(mask, "myTest.gds")
reader.apply()

# Prepare geometry
bounds = mask.getBounds()
geometry = vps.Domain()

# Substrate plane
origin = [0., 0., 0.]
normal = [0., 0., 1.]
substrate = vls.Domain(bounds, boundaryConds, gridDelta)
vls.MakeGeometry(substrate, vls.Plane(origin, normal)).apply()
geometry.insertNextLevelSetAsMaterial(substrate, vps.Material.Si)

# Insert GDS layers
layer0 = mask.layerToLevelSet(0, 0.0, 0.1, True)
geometry.insertNextLevelSetAsMaterial(layer0, vps.Material.Mask)

layer1 = mask.layerToLevelSet(1, -0.1, 0.3, True)
geometry.insertNextLevelSetAsMaterial(layer1, vps.Material.SiO2)

layer2 = mask.layerToLevelSet(2, 0.0, 0.15, True, False)
geometry.insertNextLevelSetAsMaterial(layer2, vps.Material.Si3N4)

layer3 = mask.layerToLevelSet(3, 0.0, 0.25, True)
geometry.insertNextLevelSetAsMaterial(layer3, vps.Material.Cu)

layer4 = mask.layerToLevelSet(4, 0.0, 0.4, True, False)
geometry.insertNextLevelSetAsMaterial(layer4, vps.Material.W)

layer5 = mask.layerToLevelSet(5, 0.0, 0.2, True)
geometry.insertNextLevelSetAsMaterial(layer5, vps.Material.PolySi)

# Output meshes
geometry.saveSurfaceMesh("Geometry.vtp", False)
geometry.saveVolumeMesh("Geometry")