from argparse import ArgumentParser
import numpy as np

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="testFluxes", description="Test different flux configurations."
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

vps.setNumThreads(16)
vps.Logger.setLogLevel(vps.LogLevel.INFO)

vps.Length.setUnit("um")
vps.Time.setUnit("min")

# hole geometry parameters
<<<<<<< HEAD
gridDelta = 0.018  # um
=======
gridDelta = 0.03  # um
>>>>>>> ce5b5baa2fe8e160779b04fc0c74204adfdf6bc7
extent = 1.5
holeRadius = 0.175
maskHeight = 1.2
taperAngle = 1.193

# fluxes
ionFlux = [10.0, 10.0, 10.0, 10.0, 10.0]
etchantFlux = [4.8e3, 4.5e3, 4e3, 3.5e3, 5e3]
oxygenFlux = [3e2, 8e2, 2e3, 2.5e3, 0.0]
A_O = [2, 2, 2, 1, 1]
yo2 = [0.44, 0.5, 0.56, 0.62, 0]

# etching model parameters
params = vps.SF6O2Etching.defaultParameters()
params.Substrate.A_ie = 5.0
params.Substrate.Eth_ie = 15.0

params.Substrate.A_sp = 0.0337
params.Substrate.Eth_sp = 20.0

params.Ions.exponent = 500
params.Ions.meanEnergy = 100.0
params.Ions.sigmaEnergy = 10.0
params.Ions.minAngle = np.deg2rad(85.0)
params.Ions.inflectAngle = np.deg2rad(89.0)

params.Mask.rho = params.Substrate.rho * 10.0

# simulation parameters
processDuration = 3  # min
integrationScheme = vps.IntegrationScheme.ENGQUIST_OSHER_2ND_ORDER
numberOfRaysPerPoint = int(1000)

for i in range(len(yo2)):

    # geometry setup, all units in um
    geometry = vps.Domain(
        gridDelta=gridDelta,
        xExtent=extent,
        yExtent=extent,
    )
    vps.MakeHole(
        domain=geometry,
        holeRadius=holeRadius,
        holeDepth=0.0,
        maskHeight=maskHeight,
        maskTaperAngle=taperAngle,
        holeShape=vps.HoleShape.Half,
    ).apply()

    process = vps.Process()
    process.setDomain(geometry)
    process.setMaxCoverageInitIterations(20)
    process.setCoverageDeltaThreshold(1e-4)
    process.setProcessDuration(processDuration)
    process.setIntegrationScheme(integrationScheme)
    process.setNumberOfRaysPerPoint(numberOfRaysPerPoint)

    params.ionFlux = ionFlux[i]
    params.etchantFlux = etchantFlux[i]
    params.passivationFlux = oxygenFlux[i]
    params.Passivation.A_ie = A_O[i]

    model = vps.SF6O2Etching(params)

    process.setProcessModel(model)
    process.apply()

    # save mask
    mask = geometry.getLevelSets()[0]
    mesh = vps.ls.Mesh()
    vps.ls.ToSurfaceMesh(mask, mesh).apply()
    vps.ls.VTKWriter(mesh, "mask_y{:.2f}.vtp".format(yo2[i])).apply()

    geometry.saveSurfaceMesh("hole_y{:.2f}.vtp".format(yo2[i]), True)
    geometry.saveVolumeMesh("hole_y{:.2f}".format(yo2[i]))
