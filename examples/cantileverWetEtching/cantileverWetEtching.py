# This example only works in 3D mode
import viennaps3d as vps
import viennals3d as vls

maskFileName = "cantilever_mask.gds"

# crystal surface direction
direction100 = [0.707106781187, 0.707106781187, 0.0]
direction010 = [-0.707106781187, 0.707106781187, 0.0]
# etch rates for crystal directions in um / s
# 30 % KOH at 70°C
# https://doi.org/10.1016/S0924-4247(97)01658-0
r100 = 0.797 / 60.0
r110 = 1.455 / 60.0
r111 = 0.005 / 60.0
r311 = 1.436 / 60.0
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

# set up domain
geometry = vps.Domain()
geometry.insertNextLevelSet(mask)
geometry.insertNextLevelSet(plane)
geometry.saveSurface("initialGeometry.vtp", True)

# wet etch process
model = vps.AnisotropicProcess(
    direction100=direction100,
    direction010=direction010,
    rate100=r100,
    rate110=r110,
    rate111=r111,
    rate311=r311,
    materials=[(vps.Material.Si, -1.0)],
)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(5.0 * 60.0)  # 5 minutes of etching
process.setIntegrationScheme(
    vls.lsIntegrationSchemeEnum.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

for n in range(minutes):
    # run process
    process.apply()
    geometry.saveSurface("wetEtchingSurface_" + str(n) + ".vtp", True)

geometry.saveSurface("finalGeometry.vtp", True)
