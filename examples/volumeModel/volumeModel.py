import viennaps3d as vps
from argparse import ArgumentParser

# parse config file name
parser = ArgumentParser(
    prog="volumeModel", description="Run a plasma damage process on a fin geometry."
)
parser.add_argument("filename")
args = parser.parse_args()

params = vps.ReadConfigFile(args.filename)

geometry = vps.Domain()
vps.MakeFin(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    finWidth=params["finWidth"],
    finHeight=params["finHeight"],
).apply()


# generate cell set with depth 5
geometry.generateCellSet(-5.0, False)

model = vps.PlasmaDamage(
    ionEnergy=params["ionEnergy"], meanFreePath=params["meanFreePath"], maskMaterial=-1
)

process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(0)  # apply only damage model

process.apply()

geometry.getCellSet().writeVTU("DamageModel.vtu")
