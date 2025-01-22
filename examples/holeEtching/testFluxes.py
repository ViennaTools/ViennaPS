from argparse import ArgumentParser
import numpy as np

# parse config file name and simulation dimension
parser = ArgumentParser(prog="holeEtching", description="Run a hole etching process.")
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps


vps.Logger.setLogLevel(vps.LogLevel.INFO)

# hole geometry parameters
gridDelta = 0.04  # um
xExtent = 1.2
yExtent = 1.2
holeRadius = 0.175
maskHeight = 1.2
taperAngle = 1.193

# fluxes
ionFlux = [10.0, 10.0, 10.0, 10.0, 10.0]
etchantFlux = [5.5e3, 5e3, 4e3, 3e3, 1e3]
oxygenFlux = [2e2, 3e2, 1e3, 1.5e3, 0.0]
yo2 = [0.44, 0.5, 0.56, 0.62, 0]

# etching model parameters
params = vps.SF6O2Parameters()
params.Si.rho = 5.02

params.Si.A_ie = 7.0
params.Si.Eth_ie = 15.0

params.Si.A_sp = 0.0337
params.Si.Eth_sp = 20.0

params.Passivation.A_ie = 3.0

params.Ions.exponent = 500
params.Ions.minEnergy = 100.0
params.Ions.deltaEnergy = 10.0
params.Ions.minAngle = np.deg2rad(10.0)

params.Mask.rho = 500

# simulation parameters
processDuration = 0.1  # s
integrationScheme = vps.ls.IntegrationSchemeEnum.ENGQUIST_OSHER_2ND_ORDER
numberOfRaysPerPoint = int(1000)

for i in range(len(yo2)):

    geometry = vps.Domain()
    vps.MakeHole(
        geometry,
        gridDelta,
        xExtent,
        yExtent,
        holeRadius,
        maskHeight,
        taperAngle,
        0.0,
        False,
        True,
        vps.Material.Si,
    ).apply()

    process = vps.Process()
    process.setDomain(geometry)
    process.setProcessDuration(processDuration)
    process.setIntegrationScheme(integrationScheme)
    process.setNumberOfRaysPerPoint(numberOfRaysPerPoint)

    params.ionFlux = ionFlux[i]
    params.etchantFlux = etchantFlux[i]
    params.oxygenFlux = oxygenFlux[i]
    model = vps.SF6O2Etching(params)

    process.setProcessModel(model)
    process.apply()

    geometry.saveSurfaceMesh("hole_y{:.2f}_EO2_old.vtp".format(yo2[i]))
