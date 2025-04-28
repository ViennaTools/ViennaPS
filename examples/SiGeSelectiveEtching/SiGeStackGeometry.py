# Create a SiGe stack geometry with SiO2 mask

import viennaps2d as vps
import viennals2d as vls

def CreateGeometry(paramDict: dict):
    domain = vps.Domain()

    totalHeight = paramDict["numLayers"] * paramDict["layerHeight"]
    extent = (
        2 * paramDict["lateralSpacing"]
        + paramDict["numPillars"] * paramDict["maskWidth"]
        + (paramDict["numPillars"] - 1) * paramDict["trenchWidthTop"]
    )
    bounds = [-extent / 2, extent / 2, -1, 1]
    boundaryConds = [
        vls.BoundaryConditionEnum.PERIODIC_BOUNDARY,
        vls.BoundaryConditionEnum.INFINITE_BOUNDARY,
    ]

    normal = [0, 1]
    origin = [0, 0]

    # substrate plane
    plane = vls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    vls.MakeGeometry(plane, vls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(levelSet = plane, material = vps.Material.Si, wrapLowerLevelSet = True)

    # alternating layers
    for i in range(paramDict["numLayers"]):
        origin[1] += paramDict["layerHeight"]
        plane = vls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
        vls.MakeGeometry(plane, vls.Plane(origin, normal)).apply()
        if i % 2 == 0:
            domain.insertNextLevelSetAsMaterial(plane, vps.Material.SiGe)
        else:
            domain.insertNextLevelSetAsMaterial(plane, vps.Material.Si)

    # SiO2 mask
    maskPosY = totalHeight + paramDict["maskHeight"]
    origin[1] = maskPosY
    mask = vls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    vls.MakeGeometry(mask, vls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(mask, vps.Material.SiO2)

    # mask
    maskPosY = totalHeight + paramDict["maskHeight"] + 5 * paramDict["gridDelta"]
    origin[1] = maskPosY
    etchMask = vls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    vls.MakeGeometry(etchMask, vls.Plane(origin, normal)).apply()
    domain.insertNextLevelSetAsMaterial(etchMask, vps.Material.Mask)

    # left right space
    minPoint = [
        -extent / 2 - paramDict["gridDelta"],
        totalHeight + paramDict["maskHeight"],
    ]
    maxPoint = [
        -extent / 2 + paramDict["lateralSpacing"],
        maskPosY + paramDict["gridDelta"],
    ]
    box = vls.Domain(bounds, boundaryConds, paramDict["gridDelta"])
    vls.MakeGeometry(box, vls.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    minPoint[0] = extent / 2 - paramDict["lateralSpacing"]
    maxPoint[0] = extent / 2 + paramDict["gridDelta"]
    vls.MakeGeometry(box, vls.Box(minPoint, maxPoint)).apply()
    domain.applyBooleanOperation(box, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    xpos = -extent / 2 + paramDict["lateralSpacing"] + paramDict["maskWidth"]
    for i in range(paramDict["numPillars"]):
        minPoint[0] = xpos
        maxPoint[0] = xpos + paramDict["trenchWidthTop"]
        vls.MakeGeometry(box, vls.Box(minPoint, maxPoint)).apply()
        domain.applyBooleanOperation(
            box, vls.BooleanOperationEnum.RELATIVE_COMPLEMENT
        )
        xpos += paramDict["maskWidth"] + paramDict["trenchWidthTop"]

    vps.Logger.setLogLevel(vps.LogLevel.WARNING)

    # trench etching
    direction = [0, -1, 0]
    isoVel = (
        -0.05
        * (paramDict["trenchWidthTop"] - paramDict["trenchWidthBottom"])
        / (paramDict["numLayers"] * paramDict["layerHeight"] + paramDict["maskHeight"])
    )
    processModel = vps.DirectionalProcess(direction, -1, isoVel, vps.Material.Mask)

    time = (
        paramDict["numLayers"] * paramDict["layerHeight"]
        + paramDict["maskHeight"]
        + paramDict["overEtch"]
    )
    vps.Process(domain, processModel, time).apply()

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