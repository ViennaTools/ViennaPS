import viennaps as vps

vps.setDimension(3)


def run3D():
    vps.Logger.setLogLevel(vps.LogLevel.WARNING)

    domain = vps.Domain()
    vps.MakeTrench(
        domain=domain,
        gridDelta=1.0,
        xExtent=10.0,
        yExtent=10.0,
        trenchWidth=5.0,
        trenchDepth=5.0,
        taperingAngle=0.0,
        baseHeight=0.0,
        periodicBoundary=False,
        makeMask=True,
        material=vps.Material.Si,
    ).apply()

    model = vps.DirectionalProcess(
        direction=[0.0, 0.0, -1.0],
        directionalVelocity=1.0,
        isotropicVelocity=-0.1,
        maskMaterial=vps.Material.Mask,
    )
    vps.Process(domain, model, 10.0).apply()

    model = vps.DirectionalProcess(
        direction=[0.0, 0.0, -1.0],
        directionalVelocity=1.0,
        isotropicVelocity=-0.1,
        maskMaterial=[vps.Material.Mask],
    )
    vps.Process(domain, model, 10.0).apply()

    print("3D DirectionalProcess test passed")


def testRateBased():
    vps.Logger.setLogLevel(vps.LogLevel.WARNING)

    domain = vps.Domain(
        gridDelta=0.95,
        xExtent=10.0,
        yExtent=10.0,
        boundary=vps.BoundaryType.REFLECTIVE_BOUNDARY,
    )
    vps.MakeTrench(
        domain=domain,
        trenchWidth=2.0,
        trenchDepth=0.0,
        maskHeight=5.0,
        maskMaterial=vps.Material.Mask,
        material=vps.Material.Si,
    ).apply()

    ls = domain.getLevelSets()
    print(len(ls))

    model = vps.DirectionalProcess(
        direction=[0.0, 0.0, -1.0],
        materialRates={
            vps.Material.Si: (5.0, 0.1),
            vps.Material.Mask: (0.5, 0.0),
        },
    )
    vps.Process(domain, model, 2.0).apply()

    domain.saveSurfaceMesh("DirectionalProcess3D_RateBased.vtp", addInterfaces=True)

    print("3D DirectionalProcess with rate map test passed")


if __name__ == "__main__":
    run3D()
    testRateBased()
