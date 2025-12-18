# This example only works in 3D mode
import viennaps.d3 as psd
from viennaps import BoundaryType, Material, AdvectionParameters, DiscretizationScheme

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
    BoundaryType.REFLECTIVE_BOUNDARY,
    BoundaryType.REFLECTIVE_BOUNDARY,
    BoundaryType.INFINITE_BOUNDARY,
]

gds_mask = psd.GDSGeometry(gridDelta)
gds_mask.setBoundaryConditions(boundaryConditions)
gds_mask.setBoundaryPadding(x_add, y_add)
psd.GDSReader(gds_mask, maskFileName).apply()

# convert GDS geometry to level set
mask = gds_mask.layerToLevelSet(1, 0.0, 4 * gridDelta, True, False)

# set up domain
geometry = psd.Domain()
geometry.insertNextLevelSetAsMaterial(mask, Material.Mask)

# create plane substrate under mask
psd.MakePlane(geometry, 0.0, Material.Si, True).apply()

geometry.saveSurfaceMesh("initialGeometry.vtp", True)

# wet etch process
model = psd.WetEtching(
    direction100=direction100,
    direction010=direction010,
    rate100=r100,
    rate110=r110,
    rate111=r111,
    rate311=r311,
    materialRates=[(Material.Si, -1.0)],
)

advectionParams = AdvectionParameters()
advectionParams.discretizationScheme = (
    DiscretizationScheme.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

process = psd.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(5.0 * 60.0)  # 5 minutes of etching
process.setParameters(advectionParams)

for n in range(minutes):
    # run process
    process.apply()
    geometry.saveSurfaceMesh("wetEtchingSurface_" + str(n) + ".vtp", True)

geometry.saveSurfaceMesh("finalGeometry.vtp", True)
