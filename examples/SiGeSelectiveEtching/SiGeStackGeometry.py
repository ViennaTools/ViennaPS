# Create a SiGe stack geometry with SiO2 mask

import viennaps.d2 as psd
import viennaps as ps
import viennals.d2 as lsd
from viennals import BooleanOperationEnum


def CreateGeometry(paramDict: dict) -> psd.Domain:
    domain = psd.Domain()

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
    plane = lsd.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    lsd.MakeGeometry(plane, lsd.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(
        levelSet=plane, material=ps.Material.Si, wrapLowerLevelSet=True
    )

    # alternating layers
    for i in range(paramDict["numLayers"]):
        origin[1] += paramDict["layerHeight"]
        plane = lsd.Domain(bounds, boundaryConds, paramDict["gridDelta"])
        lsd.MakeGeometry(plane, lsd.Plane(origin, normal)).apply()
        if i % 2 == 0:
            domain.insertNextLevelSetAsMaterial(plane, ps.Material.SiGe)
        else:
            domain.insertNextLevelSetAsMaterial(plane, ps.Material.Si)

    # SiO2 mask
    maskPosY = totalHeight + paramDict["maskHeight"]
    origin[1] = maskPosY
    mask = lsd.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    lsd.MakeGeometry(mask, lsd.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(mask, ps.Material.SiO2)

    # mask
    maskPosY = totalHeight + paramDict["maskHeight"] + 5 * paramDict["gridDelta"]
    origin[1] = maskPosY
    etchMask = lsd.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    lsd.MakeGeometry(etchMask, lsd.Plane(origin, normal)).apply()
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
    box = lsd.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    lsd.MakeGeometry(box, lsd.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, BooleanOperationEnum.RELATIVE_COMPLEMENT)

    minPoint[0] = extent / 2 - paramDict["lateralSpacing"]
    maxPoint[0] = extent / 2 + paramDict["gridDelta"]
    lsd.MakeGeometry(box, lsd.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, BooleanOperationEnum.RELATIVE_COMPLEMENT)

    xpos = -extent / 2 + paramDict["lateralSpacing"] + paramDict["maskWidth"]
    for i in range(paramDict["numPillars"]):
        minPoint[0] = xpos
        maxPoint[0] = xpos + paramDict["trenchWidthTop"]
        lsd.MakeGeometry(box, lsd.Box(minPoint, maxPoint)).apply()
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
    processModel = psd.DirectionalProcess(direction, -1, isoVel, ps.Material.Mask)

    time = (
        paramDict["numLayers"] * paramDict["layerHeight"]
        + paramDict["maskHeight"]
        + paramDict["overEtch"]
    )
    psd.Process(domain, processModel, time).apply()

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
