def run2D():
    import viennaps2d as vps

    # Create a 2D simulation object
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

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=vps.Material.Mask)
    vps.Process(domain, model, 10.0).apply()

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=[vps.Material.Mask])
    vps.Process(domain, model, 10.0).apply()


def run3D():
    import viennaps3d as vps

    # Create a 2D simulation object
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

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=vps.Material.Mask)
    vps.Process(domain, model, 2.0).apply()

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=[vps.Material.Mask])
    vps.Process(domain, model, 2.0).apply()


if __name__ == "__main__":
    # run2D()
    # print("2D test passed")
    run3D()
    print("3D test passed")
    print("All tests passed")
