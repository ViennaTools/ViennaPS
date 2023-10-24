import viennaps3d as vps

# parse the parameters
params = vps.psReadConfigFile("config.txt")

geometry = vps.psDomain()
vps.psMakeFin(
    psDomain=geometry,
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

process = vps.psProcess()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(0)  # apply only damage model

process.apply()

geometry.getCellSet().writeVTU("DamageModel.vtu")
