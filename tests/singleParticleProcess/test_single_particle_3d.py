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

    model = vps.SingleParticleProcess(
        rate=-1.0,
        stickingProbability=1.0,
        sourceExponent=1.0,
        maskMaterial=vps.Material.Mask,
    )
    vps.Process(domain, model, 5.0).apply()

    model = vps.SingleParticleProcess(
        rate=-1.0,
        stickingProbability=1.0,
        sourceExponent=1.0,
        maskMaterials=[vps.Material.Mask],
    )
    vps.Process(domain, model, 5.0).apply()

    print("3D SingleParticleProcess test passed")

if __name__ == "__main__":
    run3D()
