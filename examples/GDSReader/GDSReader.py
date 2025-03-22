import viennaps3d as vps

try:
    # ViennaLS Python bindings are needed for the extrusion tool
    import viennals3d as vls
except ModuleNotFoundError:
    print("ViennaLS Python module not found. Can not parse GDS file.")
    exit(1)

vps.Logger.setLogLevel(vps.LogLevel.INFO)

gridDelta = 0.01
boundaryConds = [
    vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
]

mask = vps.GDSGeometry(gridDelta)
mask.setBoundaryConditions(boundaryConds)
vps.GDSReader(mask, "mask.gds").apply()

bounds = mask.getBounds()
geometry = vps.Domain()

# substrate plane
origin = [0.0, 0.0, 0.0]
normal = [0.0, 0.0, 1.0]
plane = vls.Domain(bounds, boundaryConds, gridDelta)
vls.MakeGeometry(plane, vls.Plane(origin, normal)).apply()

geometry.insertNextLevelSet(plane)

layer0 = mask.layerToLevelSet(0, 0.0, 0.1, False)
geometry.insertNextLevelSet(layer0)

layer1 = mask.layerToLevelSet(1, -0.15, 0.45, False)
geometry.insertNextLevelSet(layer1)

geometry.saveSurfaceMesh("Geometry.vtp", True)
