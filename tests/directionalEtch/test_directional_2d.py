import viennaps2d as vps

def run2D():
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

    model = vps.DirectionalProcess([0.0, -1.0, 0.0], 1.0, -0.1, vps.Material.Mask)
    vps.Process(domain, model, 10.0).apply()

    model = vps.DirectionalProcess([0.0, -1.0, 0.0], 1.0, -0.1, [vps.Material.Mask])
    vps.Process(domain, model, 10.0).apply()

    print("2D DirectionalProcess test passed")

if __name__ == "__main__":
    run2D()
