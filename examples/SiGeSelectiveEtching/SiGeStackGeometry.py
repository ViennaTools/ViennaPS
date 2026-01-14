# Create a SiGe stack geometry with SiO2 mask

import viennaps as ps
from viennals import BooleanOperationEnum

ps.setDimension(2)


def CreateGeometry(paramDict: dict) -> ps.Domain:
    domain = ps.Domain()
    totalHeight = paramDict["numLayers"] * paramDict["layerHeight"]
    extent = (
        2 * paramDict["lateralSpacing"]
        + paramDict["numPillars"] * paramDict["maskWidth"]
        + (paramDict["numPillars"] - 1) * paramDict["trenchWidthTop"]
    )
    bounds = [-extent / 2, extent / 2, -1, 1]
    boundaryConds = [
        ps.BoundaryType.PERIODIC_BOUNDARY,
        ps.BoundaryType.INFINITE_BOUNDARY,
    ]

    normal = [0, 1]
    origin = [0, 0]

    # substrate plane
    plane = ps.ls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    ps.ls.MakeGeometry(plane, ps.ls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(
        levelSet=plane, material=ps.Material.Si, wrapLowerLevelSet=True
    )

    # alternating layers
    for i in range(paramDict["numLayers"]):
        origin[1] += paramDict["layerHeight"]
        plane = ps.ls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
        ps.ls.MakeGeometry(plane, ps.ls.Plane(origin, normal)).apply()
        if i % 2 == 0:
            domain.insertNextLevelSetAsMaterial(plane, ps.Material.SiGe)
        else:
            domain.insertNextLevelSetAsMaterial(plane, ps.Material.Si)

    # SiO2 mask
    maskPosY = totalHeight + paramDict["maskHeight"]
    origin[1] = maskPosY
    mask = ps.ls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    ps.ls.MakeGeometry(mask, ps.ls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(mask, ps.Material.SiO2)

    # mask
    maskPosY = totalHeight + paramDict["maskHeight"] + 5 * paramDict["gridDelta"]
    origin[1] = maskPosY
    etchMask = ps.ls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    ps.ls.MakeGeometry(etchMask, ps.ls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(etchMask, ps.Material.Mask)

    # left right space
    minPoint = [
        -extent / 2 - paramDict["gridDelta"],
        totalHeight + paramDict["maskHeight"],
    ]
    maxPoint = [
        -extent / 2 + paramDict["lateralSpacing"],
        maskPosY + paramDict["gridDelta"],
    ]
    box = ps.ls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    ps.ls.MakeGeometry(box, ps.ls.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, BooleanOperationEnum.RELATIVE_COMPLEMENT)

    minPoint[0] = extent / 2 - paramDict["lateralSpacing"]
    maxPoint[0] = extent / 2 + paramDict["gridDelta"]
    ps.ls.MakeGeometry(box, ps.ls.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, BooleanOperationEnum.RELATIVE_COMPLEMENT)

    xpos = -extent / 2 + paramDict["lateralSpacing"] + paramDict["maskWidth"]
    for i in range(paramDict["numPillars"]):
        minPoint[0] = xpos
        maxPoint[0] = xpos + paramDict["trenchWidthTop"]
        ps.ls.MakeGeometry(box, ps.ls.Box(minPoint, maxPoint)).apply()
        domain.applyBooleanOperation(box, BooleanOperationEnum.RELATIVE_COMPLEMENT)
        xpos += paramDict["maskWidth"] + paramDict["trenchWidthTop"]

    ps.Logger.setLogLevel(ps.LogLevel.WARNING)

    # trench etching
    direction = [0, -1, 0]
    isoVel = (
        -0.05
        * (paramDict["trenchWidthTop"] - paramDict["trenchWidthBottom"])
        / (paramDict["numLayers"] * paramDict["layerHeight"] + paramDict["maskHeight"])
    )
    processModel = ps.DirectionalProcess(direction, -1, isoVel, ps.Material.Mask)

    time = (
        paramDict["numLayers"] * paramDict["layerHeight"]
        + paramDict["maskHeight"]
        + paramDict["overEtch"]
    )
    ps.Process(domain, processModel, time).apply()

    # remove trench etching mask
    domain.removeTopLevelSet()
    return domain


if __name__ == "__main__":
    # test
    paramDict = {
        "numPillars": 3,
        "numLayers": 10,
        "layerHeight": 1.0,
        "maskWidth": 1.0,
        "maskHeight": 1.0,
        "trenchWidthTop": 2.0,
        "trenchWidthBottom": 1.5,
        "overEtch": 2.0,
        "lateralSpacing": 3.0,
        "periodicBoundary": False,
        "gridDelta": 0.1,
    }
    domain = CreateGeometry(paramDict)
    domain.saveVolumeMesh("SiGeStackGeometry")
