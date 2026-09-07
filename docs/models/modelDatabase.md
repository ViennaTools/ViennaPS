---
layout: default
title: Model Database
parent: Process Models
nav_order: 3
---

# Model Database
{: .fs-9 .fw-500}

The repository's `modeldb/` directory contains public CSV parameters for
[ion implantation]({% link models/prebuilt/ionImplantation.md %}), damage, and
[thermal annealing]({% link models/prebuilt/anneal.md %}). The data are generic
simulation- and literature-based parameters. Check the table headers and
provenance before selecting them for a particular process.

## Available data

| Directory | Coverage in this checkout |
|-----------|---------------------------|
| `implant/` | Boron and phosphorus in crystalline silicon; dual-Pearson IV profiles at 8, 10, 30, 50, 100, 130, and 200 keV, with 7° tilt and 22° twist. |
| `damage/` | Boron and phosphorus damage profiles in silicon. |
| `anneal/annealing.csv` | Intrinsic diffusivities for B, P, Al, Ga, In, As, Sb, and Bi in silicon; additional solubility and effective damage-driven TED parameters for B and P. |

Implant tables use nanometers, keV, degrees, and ions/cm². Anneal table
diffusivity prefactors use nm²/s and activation energies use eV; solubility
prefactors use cm⁻³. The C++ anneal lookup converts length-dependent parameters
to the requested geometry unit.

The public tables interpolate within their supported ranges; they do not
provide extrapolation to arbitrary energies or implant geometries. Additional
anneal diffusivity rows do not imply matching implant or damage tables.

## Configure the database location

Point to the directory containing `implant`, `damage`, and `anneal`:

```c++
#include <psModelDb.hpp>

viennaps::setModelDbRoot("/path/to/ViennaPS/modeldb");
```

```python
import viennaps as vps

vps.setModelDbRoot("/path/to/ViennaPS/modeldb")
```

Alternatively, set the `VIENNAPS_MODELDB_ROOT` environment variable and call
`initModelDbRoot()` (`vps.initModelDbRoot()` in Python). Initialization first
uses the `VIENNAPS_MODELDB_DIR` compile-time definition if the root is empty,
then applies a nonempty environment override. `getModelDbRoot()` returns the
configured path. The C++ implantation example defines a source-tree default.
For a separately installed application, configure a path to your own copy of
the database rather than assuming a checkout is available.

## C++ table-based implant setup

With a cell-set domain prepared as in the implantation guide:

```c++
#include <models/psImplantSetup.hpp>

viennaps::TableImplantRecipe<double> recipe;
recipe.species = "B";
recipe.material = "Si";
recipe.substrateType = "crystalline";
recipe.energyKeV = 10.0;
recipe.tiltDeg = 7.0;
recipe.rotationDeg = 22.0;
recipe.doseCm2 = 1e13;
recipe.screenThickness = 0.0;

auto setup = viennaps::makeTableImplant<double, 2>(recipe);
auto model = viennaps::SmartPointer<viennaps::IonImplantation<double, 2>>::New();
viennaps::applyImplantSetup(*model, setup);
viennaps::Process<double, 2>(domain, model, 0.0).apply();
```

The helper assigns species-specific field labels such as `B_total`, `B_damage`,
and `B_damage_last`. `dopantFields(species)` also supplies matching active and
defect field names for annealing.

The setup helpers shown above are C++ APIs. Python exposes analytical profiles,
`ImplantTableModel`, `DamageTableModel`, and the `IonImplantation`/`Anneal`
setters. For example, use an explicit table path to construct the dopant profile:

```python
profile = vps.ImplantTableModel(
    vps.getModelDbRoot() + "/implant/boron_in_silicon_crystalline.csv",
    "B", "Si", "crystalline", 10.0, 7.0, 22.0, 1e13, 0.0)
implant = vps.IonImplantation()
implant.setImplantModel(profile)
implant.setDose(1e13)
implant.setTiltAngle(7.0)
implant.setLengthUnit(1e-7)
```

See the Python scripts in the
[implantation example](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionImplantation)
for a complete table-based workflow with damage and annealing.

## Custom data and lookup failures

Missing database paths, unsupported species/materials, and out-of-range
conditions produce model-data errors. Use conditions covered by the tables,
provide custom CSV files through the recipe's `tableFileName`, or configure
analytical profile moments and anneal parameters directly. Supplying an implant
profile alone does not supply a damage model for a defect-coupled anneal.

The C++ helpers report lookup failures as `viennaps::modeldb::ModelDbError`.
`runWithModelDbErrors(...)` is available for command-line programs that want a
formatted diagnostic and exit code 2.

See the [database files](https://github.com/ViennaTools/ViennaPS/tree/master/modeldb)
and [anneal provenance](https://github.com/ViennaTools/ViennaPS/blob/master/modeldb/anneal/PROVENANCE.md)
for the parameter sources and table formats.
