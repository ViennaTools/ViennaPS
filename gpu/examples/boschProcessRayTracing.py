#####################################################################
# 3D Bosch process simulation using ray tracing
# Replicates the results from:
#   Ertl and Selberherr https://doi.org/10.1016/j.mee.2009.05.011
# Execute: 
#   python boschProcessRayTracing.py -D 3 configBoschRayTracing.txt
#####################################################################

from argparse import ArgumentParser
import viennaps3d as vps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="boschProcess",
    description="Run a Bosch process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

context = vps.gpu.Context()
context.create()

params = vps.ReadConfigFile(args.filename)

# print only error output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

# Map the shape string to the corresponding vps.HoleShape enum
shape_map = {
    "Full": vps.HoleShape.Full,
    "Half": vps.HoleShape.Half,
    "Quarter": vps.HoleShape.Quarter,
}

hole_shape_str = params.get("holeShape", "Full").strip()

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
    holeShape=shape_map[hole_shape_str],
).apply()

depoModel = vps.gpu.MultiParticleProcess()
depoModel.addNeutralParticle(params["stickingDep"])
depoModel.addIonParticle(params["ionSourceExponent"])

etchModel = vps.gpu.MultiParticleProcess()
materialStickingEtch = {vps.Material.Si: params["stickingEtchSubs"],
                        vps.Material.Mask: params["stickingEtchMask"],
                        vps.Material.Polymer: params["stickingEtchPoly"]}
etchModel.addNeutralParticle(materialStickingEtch, defaultStickingProbability=1.0)
etchModel.addIonParticle(params["ionSourceExponent"])

# Conversion factors
A2c = 1e-8  # 1 angstrom = 1e-8 cm
c2u = 1e4  # 1 cm = 1e4 micrometers

# Rate calculations for the depo model
alphaDep = params["alphaDep"] * (A2c ** 3)  # [cm^3/atom]
ionDepRate = alphaDep * params["Flux_ionD"] * c2u  # [um/s]
betaDep = params["betaDep"] * (A2c ** 3)  # [cm^3/atom]
neuDepRate = betaDep * params["Flux_neuD"] * c2u  # [um/s]
# Print the result
print(f"Deposition cycle:")
print(f"Poly deposit rate (ion, neutral): {ionDepRate:.3e} μm/s, {neuDepRate:.3e} μm/s")

# Rate calculations for the etch model
alphaPoly = params["alphaPoly"] * (A2c ** 3)  # [cm^3/atom]
alphaSubs = params["alphaSubs"] * (A2c ** 3)  # [cm^3/atom]
alphaMask = params["alphaMask"] * (A2c ** 3)  # [cm^3/atom]
ionEtchRatePoly = alphaPoly * params["Flux_ionE"] * c2u  # [um/s]
ionEtchRateSubs = alphaSubs * params["Flux_ionE"] * c2u  # [um/s]
ionEtchRateMask = alphaMask * params["Flux_ionE"] * c2u  # [um/s]
betaPoly = params["betaPoly"] * (A2c ** 3)  # [cm^3/atom]
betaSubs = params["betaSubs"] * (A2c ** 3)  # [cm^3/atom]
betaMask = params["betaMask"] * (A2c ** 3)  # [cm^3/atom]
neuEtchRatePoly = betaPoly * params["Flux_neuE"] * c2u  # [um/s]
neuEtchRateSubs = betaSubs * params["Flux_neuE"] * c2u  # [um/s]
neuEtchRateMask = betaMask * params["Flux_neuE"] * c2u  # [um/s]
# Print the result
print(f"Etching cycle:")
print(f"Mask etching rate (ion, neutral): {ionEtchRateMask:.3e} μm/s, {neuEtchRateMask:.3e} μm/s")
print(f"Poly etching rate (ion, neutral): {ionEtchRatePoly:.3e} μm/s, {neuEtchRatePoly:.3e} μm/s")
print(f"Subs etching rate (ion, neutral): {ionEtchRateSubs:.3e} μm/s, {neuEtchRateSubs:.3e} μm/s")


def rateFunctionDep(fluxes, material):
    rate = fluxes[1] * ionDepRate + fluxes[0] * neuDepRate
    return rate


# Custom rate function for the etch model
def rateFunctionEtch(fluxes, material):
    rate = 0.0
    if material == vps.Material.Mask:
        rate = fluxes[1] * ionEtchRateMask + fluxes[0] * neuEtchRateMask
    if material == vps.Material.Polymer:
        rate = fluxes[1] * ionEtchRatePoly + fluxes[0] * neuEtchRatePoly
    if material == vps.Material.Si:
        rate = fluxes[1] * ionEtchRateSubs + fluxes[0] * neuEtchRateSubs
    return rate


depoModel.setRateFunction(rateFunctionDep)
etchModel.setRateFunction(rateFunctionEtch)

depoProcess = vps.gpu.Process(context, geometry, depoModel, params["depTime"])
depoProcess.setTimeStepRatio(0.2)
etchProcess = vps.gpu.Process(context, geometry, etchModel, params["etchTime"])
etchProcess.setTimeStepRatio(0.2)

geometry.saveSurfaceMesh("initial.vtp")
geometry.saveVolumeMesh("initial")

geometry.duplicateTopLevelSet(vps.Material.Polymer)

numCycles = int(params["numCycles"])
print(f"---------------------------")
print(f"Starting {numCycles} cycles")
for i in range(numCycles):
    print(f"Cycle {i + 1}")
    # geometry.saveLevelSetMesh(filename=f"run_{2*i}", width=3)
    geometry.saveSurfaceMesh(f"run_{2 * i}.vtp")
    # geometry.saveVolumeMesh(f"run_{2*i}")
    print(f"  - Deposition -")
    depoProcess.apply()
    # geometry.saveLevelSetMesh(filename=f"run_{2*i+1}", width=3)
    geometry.saveSurfaceMesh(f"run_{2 * i + 1}.vtp")
    # geometry.saveVolumeMesh(f"run_{2*i+1}")
    print(f"  - Etching -")
    etchProcess.apply()

geometry.saveSurfaceMesh(f"final.vtp")
geometry.saveVolumeMesh(f"final")
