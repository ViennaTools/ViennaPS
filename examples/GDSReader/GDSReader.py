import viennaps.d3 as psd
import viennaps as ps
import viennals.d3 as vls

ps.Logger.setLogLevel(ps.LogLevel.DEBUG)

gridDelta = 0.01
exposureDelta = 0.005
forwardSigma = 5.0
backSigma = 50.0

boundaryConds = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]

mask = psd.GDSGeometry(gridDelta, boundaryConds)
mask.addBlur([forwardSigma, backSigma], [0.8, 0.2], 0.5, exposureDelta)

reader = psd.GDSReader(mask, "myTest.gds")
reader.apply()

# Prepare geometry
bounds = mask.getBounds()
geometry = psd.Domain()

# Substrate plane
origin = [0.0, 0.0, 0.0]
normal = [0.0, 0.0, 1.0]
substrate = vls.Domain(bounds, boundaryConds, gridDelta)
vls.MakeGeometry(substrate, vls.Plane(origin, normal)).apply()
geometry.insertNextLevelSetAsMaterial(substrate, ps.Material.Si)

# Insert GDS layers
layer0 = mask.layerToLevelSet(0, 0.0, 0.1, True)
geometry.insertNextLevelSetAsMaterial(layer0, ps.Material.Mask)

layer1 = mask.layerToLevelSet(1, -0.1, 0.3, True)
geometry.insertNextLevelSetAsMaterial(layer1, ps.Material.SiO2)

layer2 = mask.layerToLevelSet(2, 0.0, 0.15, True, False)
geometry.insertNextLevelSetAsMaterial(layer2, ps.Material.Si3N4)

layer3 = mask.layerToLevelSet(3, 0.0, 0.25, True)
geometry.insertNextLevelSetAsMaterial(layer3, ps.Material.Cu)

layer4 = mask.layerToLevelSet(4, 0.0, 0.4, True, False)
geometry.insertNextLevelSetAsMaterial(layer4, ps.Material.W)

layer5 = mask.layerToLevelSet(5, 0.0, 0.2, True)
geometry.insertNextLevelSetAsMaterial(layer5, ps.Material.PolySi)

# Output meshes
geometry.saveSurfaceMesh("Geometry.vtp", False)
geometry.saveVolumeMesh("Geometry")
