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
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

params = vps.ReadConfigFile(args.filename)

geometry = vps.Domain(
    gridDelta=params["gridDelta"], xExtent=params["xExtent"], yExtent=params["yExtent"]
)
vps.MakePlane(
    domain=geometry, height=0.0, material=vps.Material.Si, addToExisting=False
).apply()

fin = vps.ls.Domain(geometry.getLevelSets()[-1])

if args.dim == 3:
    vps.ls.MakeGeometry(
        fin,
        vps.ls.Box(
            [
                -params["finWidth"] / 2.0,
                -params["finLength"] / 2.0,
                -params["gridDelta"],
            ],
            [params["finWidth"] / 2.0, params["finLength"] / 2.0, params["finHeight"]],
        ),
    ).apply()
else:
    vps.ls.MakeGeometry(
        fin,
        vps.ls.Box(
            [
                -params["finWidth"] / 2.0,
                -params["gridDelta"],
            ],
            [params["finWidth"] / 2.0, params["finHeight"]],
        ),
    ).apply()

geometry.applyBooleanOperation(fin, vps.ls.BooleanOperationEnum.UNION)

geometry.saveVolumeMesh("fin")

vps.MakePlane(
    domain=geometry,
    height=params["oxideHeight"],
    material=vps.Material.SiO2,
    addToExisting=True,
).apply()

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
    vps.ls.IntegrationSchemeEnum.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

geometry.saveVolumeMesh("initial")

process.apply()

geometry.saveVolumeMesh("final")
