from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="selectiveEpitaxy",
    description="Run a selective epitaxial growth process.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
    import viennals2d as vls
else:
    print("Running 3D simulation.")
    import viennaps3d as vps
    import viennals3d as vls

params = vps.ReadConfigFile(args.filename)

geometry = vps.Domain()
vps.MakePlane(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    height=0.0,
    periodicBoundary=False,
    material=vps.Material.Mask,
).apply()

fin = vls.lsDomain(geometry.getLevelSets()[-1])

if args.dim == 3:
    vls.lsMakeGeometry(
        fin,
        vls.lsBox(
            [
                -params["finWidth"] / 2.0,
                -params["finLength"] / 2.0,
                -params["gridDelta"],
            ],
            [params["finWidth"] / 2.0, params["finLength"] / 2.0, params["finHeight"]],
        ),
    ).apply()
else:
    vls.lsMakeGeometry(
        fin,
        vls.lsBox(
            [
                -params["finWidth"] / 2.0,
                -params["gridDelta"],
            ],
            [params["finWidth"] / 2.0, params["finHeight"]],
        ),
    ).apply()

geometry.insertNextLevelSetAsMaterial(fin, vps.Material.Si)

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(vps.Material.SiGe)

model = vps.AnisotropicProcess(
    materials=[
        (vps.Material.Si, params["epitaxyRate"]),
        (vps.Material.SiGe, params["epitaxyRate"]),
    ],
)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.setIntegrationScheme(
    vls.lsIntegrationSchemeEnum.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

geometry.printSurface("initial.vtp")

process.apply()

geometry.printSurface("final.vtp")
vps.WriteVisualizationMesh(geometry, "final").apply()
