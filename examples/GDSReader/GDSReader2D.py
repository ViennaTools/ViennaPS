import viennaps2d as vps
try:
    # ViennaLS Python bindings are needed for the extrusion tool
    import viennals2d as vls
except ModuleNotFoundError:
    print("ViennaLS Python module not found. Can not parse GDS file.")
    exit(1)

# Set log level
vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

# Parameters
gridDelta = 0.01
exposureDelta = 0.005
forwardSigma = 5.0
backsSigma = 50.0

boundaryConds = [
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
    vps.ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
]

# Create GDS geometry object
mask = vps.GDSGeometry(gridDelta, boundaryConds)

# Add blur parameters
mask.addBlur(
    sigmas=[forwardSigma, backsSigma],
    weights=[0.8, 0.2],
    threshold=0.5,
    delta=exposureDelta
)

# Load GDS file
reader = vps.GDSReader(mask, "myTest.gds")
reader.apply()

# Export unblurred mask layer
maskLayer = mask.layerToLevelSet(0, blurLayer=False)
mesh = vls.Mesh()
vls.ToSurfaceMesh(maskLayer, mesh).apply()
vls.VTKWriter(mesh, "maskLayer.vtp").apply()

# Export blurred mask layer
blurredLayer = mask.layerToLevelSet(0, blurLayer=True)
vls.ToSurfaceMesh(blurredLayer, mesh).apply()
vls.VTKWriter(mesh, "blurredLayer.vtp").apply()