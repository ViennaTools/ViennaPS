import viennaps3d as vps

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

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=vps.Material.Mask)
    vps.Process(domain, model, 2.0).apply()

    model = vps.IsotropicProcess(rate=-1.0, maskMaterial=[vps.Material.Mask])
    vps.Process(domain, model, 2.0).apply()

    print("3D IsotropicProcess test passed")

if __name__ == "__main__":
    run3D()
