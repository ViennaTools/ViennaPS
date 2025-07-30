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
vps.MakeFin(
    domain=geometry,
    finWidth=params["finWidth"],
    finHeight=params["finHeight"],
).apply()

vps.MakePlane(
    domain=geometry,
    height=params["oxideHeight"],
    material=vps.Material.SiO2,
    addToExisting=True,
).apply()

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(vps.Material.SiGe)

model = vps.SelectiveEpitaxy(
    materialRates=[
        (vps.Material.Si, params["epitaxyRate"]),
        (vps.Material.SiGe, params["epitaxyRate"]),
    ],
)

advectionParams = vps.AdvectionParameters()
advectionParams.integrationScheme = (
    vps.IntegrationScheme.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

process = vps.Process(geometry, model, params["processTime"])
process.setAdvectionParameters(advectionParams)

geometry.saveVolumeMesh("initial_fin")

process.apply()

geometry.saveVolumeMesh("final_fin")
