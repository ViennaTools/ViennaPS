"""Import a structure of material IDs from an OpenVDB file and deposit a liner.

Reading .vdb files needs NumPy and, for Blosc-compressed files, blosc:

    pip install numpy blosc
    python VDBImport.py -D 3 config.txt
"""

from argparse import ArgumentParser

import viennals as vls
import viennals.vdb
import viennaps as ps

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="VDBImport",
    description="Import a structure from an OpenVDB file and deposit a liner on it.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

ps.setDimension(args.dim)
params = ps.readConfigFile(args.filename)

# material of each ID; built-in material names map to ViennaPS materials,
# other names become new materials
entries = params["materials"]
materials = {}
for entry in [entries] if isinstance(entries, str) else entries:
    label, name = entry.split(":")
    materials[int(label)] = getattr(ps.Material, name.strip(), name.strip())

grid = vls.vdb.readGrids(params["vdbFile"])[0]
levelSets = vls.vdb.levelSetsFromLabels(
    grid, args.dim, params[f"axes{args.dim}D"], ambient=int(params["ambient"])
)

domain = ps.Domain()
for label, levelSet in levelSets:
    material = materials.get(label, f"Material{label}")
    print(f"material ID {label}: {material}")
    # each level set already encloses the materials below it
    domain.insertNextLevelSetAsMaterial(levelSet, material, False)
domain.saveVolumeMesh("initial")

domain.duplicateTopLevelSet(ps.Material.SiO2)
process = ps.Process(domain, ps.IsotropicProcess(rate=1.0))
process.setProcessDuration(params["linerThickness"])
process.apply()

domain.saveVolumeMesh("final")
