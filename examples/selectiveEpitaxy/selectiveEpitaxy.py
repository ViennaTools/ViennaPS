from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="selectiveEpitaxy",
    description="Run a selective epitaxial growth process.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

import viennaps as ps

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    psd = ps.d2
else:
    print("Running 3D simulation.")
    psd = ps.d3

params = ps.readConfigFile(args.filename)

geometry = psd.Domain(
    gridDelta=params["gridDelta"], xExtent=params["xExtent"], yExtent=params["yExtent"]
)
psd.MakeFin(
    domain=geometry,
    finWidth=params["finWidth"],
    finHeight=params["finHeight"],
).apply()

psd.MakePlane(
    domain=geometry,
    height=params["oxideHeight"],
    material=ps.Material.SiO2,
    addToExisting=True,
).apply()

# copy top layer to capture deposition
geometry.duplicateTopLevelSet(ps.Material.SiGe)

model = psd.SelectiveEpitaxy(
    materialRates=[
        (ps.Material.Si, params["epitaxyRate"]),
        (ps.Material.SiGe, params["epitaxyRate"]),
    ],
    rate111=params["R111"],
    rate100=params["R100"],
)

advectionParams = ps.AdvectionParameters()
advectionParams.integrationScheme = (
    ps.IntegrationScheme.STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER
)

process = psd.Process(geometry, model, params["processTime"])
process.setAdvectionParameters(advectionParams)

geometry.saveVolumeMesh("initial_fin")

process.apply()

geometry.saveVolumeMesh("final_fin")
