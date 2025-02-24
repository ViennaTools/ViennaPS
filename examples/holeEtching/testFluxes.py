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
gridDelta = 0.025  # um
xExtent = 1.0
yExtent = 1.0
holeRadius = 0.175
maskHeight = 1.2
taperAngle = 1.193

# fluxes
ionFlux = [10.0, 10.0, 10.0, 10.0, 10.0]
etchantFlux = [4.8e3, 4.5e3, 4e3, 3.5e3, 4e3]
oxygenFlux = [3e2, 8e2, 2e3, 2.5e3, 0.0]
A_O = [2, 2, 2, 1, 1]
yo2 = [0.44, 0.5, 0.56, 0.62, 0]

# etching model parameters
params = vps.SF6O2Parameters()
params.Si.A_ie = 5.0
params.Si.Eth_ie = 15.0

params.Si.A_sp = 0.0337
params.Si.Eth_sp = 20.0

params.Ions.exponent = 500
params.Ions.meanEnergy = 100.0
params.Ions.sigmaEnergy = 10.0
params.Ions.minAngle = np.deg2rad(85.0)
params.Ions.inflectAngle = np.deg2rad(89.0)

params.Mask.rho = params.Si.rho * 10.0

# simulation parameters
processDuration = 3  # min
integrationScheme = vps.IntegrationScheme.ENGQUIST_OSHER_2ND_ORDER
numberOfRaysPerPoint = int(1000)

for i in range(len(yo2)):

    # geometry setup, all units in um
    geometry = vps.Domain(
        gridDelta=params["gridDelta"],
        xExtent=params["xExtent"],
        yExtent=params["yExtent"],
    )
    vps.MakeHole(
        domain=geometry,
        holeRadius=params["holeRadius"],
        holeDepth=0.0,
        maskHeight=params["maskHeight"],
        maskTaperAngle=params["taperAngle"],
        holeShape=vps.HoleShape.Half,
    ).apply()

    process = vps.Process()
    process.setDomain(geometry)
    process.setMaxCoverageInitIterations(20)
    process.setProcessDuration(processDuration)
    process.setIntegrationScheme(integrationScheme)
    process.setNumberOfRaysPerPoint(numberOfRaysPerPoint)

    params.ionFlux = ionFlux[i]
    params.etchantFlux = etchantFlux[i]
    params.oxygenFlux = oxygenFlux[i]
    params.Passivation.A_ie = A_O[i]

    model = vps.SF6O2Etching(params)

    process.setProcessModel(model)
    process.apply()

    geometry.saveSurfaceMesh("hole_y{:.2f}_EO2.vtp".format(yo2[i]), True)
