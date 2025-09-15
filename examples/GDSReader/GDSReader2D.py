import viennaps as ps

# Set log level
ps.Logger.setLogLevel(ps.LogLevel.DEBUG)

# Parameters
gridDelta = 0.01
exposureDelta = 0.005
forwardSigma = 5.0
backsSigma = 50.0

boundaryConds = [
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
    ps.BoundaryType.REFLECTIVE_BOUNDARY,
]

# Create GDS geometry object
mask = ps.GDSGeometry(gridDelta, boundaryConds)

# Add blur parameters
mask.addBlur(
    sigmas=[forwardSigma, backsSigma],
    weights=[0.8, 0.2],
    threshold=0.5,
    delta=exposureDelta,
)

# Load GDS file
reader = ps.GDSReader(mask, "myTest.gds")
reader.apply()

# Export unblurred mask layer
maskLayer = mask.layerToLevelSet(0, blurLayer=False)
mesh = ps.ls.Mesh()
ps.ls.d2.ToSurfaceMesh(maskLayer, mesh).apply()
ps.ls.VTKWriter(mesh, "maskLayer.vtp").apply()

# Export blurred mask layer
blurredLayer = mask.layerToLevelSet(0, blurLayer=True)
ps.ls.d2.ToSurfaceMesh(blurredLayer, mesh).apply()
ps.ls.VTKWriter(mesh, "blurredLayer.vtp").apply()
