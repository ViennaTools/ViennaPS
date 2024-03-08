import viennaps2d as vp2d

# Create a 2D simulation object
domain2D = vp2d.Domain()
vp2d.MakeTrench(
    domain=domain2D,
    gridDelta=1.0,
    xExtent=10.0,
    yExtent=10.0,
    trenchWidth=5.0,
    trenchDepth=5.0,
    taperingAngle=0.0,
    baseHeight=0.0,
    periodicBoundary=False,
    makeMask=True,
    material=vp2d.Material.Si,
).apply()

model = vp2d.DirectionalEtching([0.0, -1.0, 0.0], 1.0, -0.1, vp2d.Material.Mask)
vp2d.Process(domain2D, model, 10.0).apply()

model = vp2d.DirectionalEtching([0.0, -1.0, 0.0], 1.0, -0.1, [vp2d.Material.Mask])
vp2d.Process(domain2D, model, 10.0).apply()
