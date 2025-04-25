import viennaps2d as vps
import viennals2d as vls

def make_initial_geometry(domain, params):
    extent = params.getExtent()
    x_max = extent[0] / 2. if params.halfGeometry else extent[0]
    bounds = [0., x_max, -1., 1.]
    bcs = [
        vls.BoundaryConditionEnum.PERIODIC_BOUNDARY if params.periodicBoundary else vls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY
    ]

    origin = [0., 0.]
    normal = [0., 1.]
    Si = vls.Domain(bounds, bcs, params.gridDelta)

    # Add Si substrate at y = 0
    vls.MakeGeometry(Si, vls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(Si, vps.Material.Si)

    # Add alternating Si and SiGe layers
    for i in range(params.numLayers):
        origin[1] += params.layerHeight
        material = vps.Material.SiGe if i % 2 == 0 else vps.Material.Si
        SiGe = vls.Domain(bounds, bcs, params.gridDelta)
        vls.MakeGeometry(SiGe, vls.Plane(origin, normal)).apply()
        if i % 2 == 0:
            domain.insertNextLevelSetAsMaterial(levelSet=SiGe, material=material)
        else:
            domain.insertNextLevelSetAsMaterial(levelSet=SiGe, material=material)

    # Add SiO2 mask on top
    totalHeight = params.numLayers * params.layerHeight
    origin[1] = totalHeight + params.maskHeight
    SiO2 = vls.Domain(bounds, bcs, params.gridDelta)
    vls.MakeGeometry(SiO2, vls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(levelSet=SiO2, material=vps.Material.SiO2)