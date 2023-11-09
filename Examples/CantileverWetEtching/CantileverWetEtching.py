# This example only works in 3D mode
import viennaps3d as vps
import viennals3d as vls

maskFileName = "cantilever_mask.gds"

minutes = int(120 / 5)  # total etch time (2 hours)
x_add = 50.0  # add space to domain boundary
y_add = 50.0
gridDelta = 5.0  # um

# read GDS mask file
boundaryConditions = [
    vls.lsBoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vls.lsBoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vls.lsBoundaryConditionEnum.INFINITE_BOUNDARY,
]

gds_mask = vps.GDSGeometry(gridDelta)
gds_mask.setBoundaryConditions(boundaryConditions)
gds_mask.setBoundaryPadding(x_add, y_add)
vps.GDSReader(gds_mask, maskFileName).apply()

# convert GDS geometry to level set
mask = gds_mask.layerToLevelSet(1, 0.0, 4 * gridDelta, True)

# create plane geometry as substrate
bounds = gds_mask.getBounds()
plane = vls.lsDomain(bounds, boundaryConditions, gridDelta)
vls.lsMakeGeometry(plane, vls.lsPlane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])).apply()

# combine geometries
geometry = vps.Domain()
geometry.insertNextLevelSet(mask)
geometry.insertNextLevelSet(plane)
geometry.printSurface("InitialGeometry.vtp", True)

# wet etch process
model = vps.WetEtching(0)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(5.0 * 60.0)  # 5 minutes of etching
process.setPrintTimeInterval(-1.0)
process.setIntegrationScheme(
    vls.lsIntegrationSchemeEnum.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

for n in range(minutes):
    process.apply()
    # run process
    geometry.printSurface("WetEtchingSurface_" + str(n) + ".vtp", True)

geometry.printSurface("FinalGeometry.vtp", True)
