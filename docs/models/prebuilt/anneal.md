---
layout: default
title: Thermal Annealing
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 17
---

# Thermal Annealing
{: .fs-9 .fw-500}

```c++
#include <models/psAnneal.hpp>
```

`Anneal` evolves dopant concentration and optional activation and defect fields
on a ViennaCS cell set. It supports constant or Arrhenius diffusivity,
isothermal steps, temperature ramps, and material-specific diffusion barriers.
The domain must already have a cell set containing the species field, for
example from [ion implantation]({% link models/prebuilt/ionImplantation.md %}).

Set the anneal duration or schedule on the model and use **zero duration** for
the enclosing `Process`. The model updates volume fields without moving the
level sets.

## Example: diffuse an implanted profile

These snippets continue from the implantation example, with `B_total` stored
in nm⁻³. The constant diffusivity is illustrative.

```c++
auto anneal = ps::SmartPointer<ps::Anneal<double, 2>>::New();
anneal->setSpeciesLabel("B_total");
anneal->setTemperature(1273.15); // kelvin
anneal->setDuration(30.0);      // seconds
anneal->setDiffusionCoefficient(1.0); // nm²/s
anneal->setMode(ps::AnnealMode::GaussSeidel);
anneal->setDiffusionMaterials({ps::Material::Si});
anneal->setBlockingMaterials(
    {ps::Material::Air, ps::Material::Mask, ps::Material::SiO2});
ps::Process<double, 2>(domain, anneal, 0.0).apply();
domain->getCellSet()->writeVTU("post_anneal");
```

```python
anneal = vps.Anneal()
anneal.setSpeciesLabel("B_total")
anneal.setTemperature(1273.15) # kelvin
anneal.setDuration(30.0)      # seconds
anneal.setDiffusionCoefficient(1.0) # nm²/s
anneal.setMode(vps.AnnealMode.GaussSeidel)
anneal.setDiffusionMaterials([vps.Material.Si])
anneal.setBlockingMaterials(
    [vps.Material.Air, vps.Material.Mask, vps.Material.SiO2])
vps.Process(domain, anneal, 0.0).apply()
domain.getCellSet().writeVTU("post_anneal")
```

## Temperature schedules and solver controls

`setTemperatureSchedule(durations, temperatures)` accepts either:

* N durations and N temperatures for N isothermal steps, or
* N durations and N+1 temperatures for linear ramps between endpoints.

For a ramp-up, soak, and ramp-down schedule:

```python
anneal.setTemperatureSchedule(
    [9.0, 5.0, 9.0],
    [873.15, 1323.15, 1323.15, 873.15])
```

Use `clearTemperatureSchedule()` before returning to a single isothermal
duration. Steps can also be appended with `addIsothermalStep` and `addRampStep`.

| Method | Purpose |
|--------|---------|
| `setArrheniusParameters(D0, Ea)` | Temperature-dependent diffusivity; D0 in geometry unit²/s and Ea in eV. |
| `setMode(mode)` | Choose `AnnealMode::Explicit` or `AnnealMode::GaussSeidel`. |
| `setTimeStep(dt)` | Time step in seconds; nonpositive values request automatic selection. |
| `setStabilityFactor(factor)` | Stability factor for explicit time stepping. |
| `setImplicitSolverOptions(maxIterations, relativeTolerance, relaxation)` | Iteration controls for Gauss–Seidel mode. |
| `setDiffusionMaterials(materials)` | Restrict diffusion to selected materials; an empty list allows all materials. |
| `setBlockingMaterials(materials)` | Select materials that block diffusion. |

## Activation and defects

`enableSolidActivation()` enables a solubility-limited active concentration
field. Name it with `setActiveLabel(...)` and configure the solubility with
`setSolidSolubilityArrhenius(C0, Ea)`, using concentrations consistent with the
cell-set fields and energy in eV.

The model also exposes damage-dependent activation, interstitial/vacancy
diffusion and reactions, transient enhanced diffusion (TED), defect clustering,
and interface trapping/segregation. These options require corresponding
parameters and fields. Match `setDamageLabels(...)` to the implant's cumulative
and most-recent damage labels when coupling an implant and anneal.

The [public model database]({% link models/modelDatabase.md %}) supplies
literature-based diffusion and selected activation/TED parameters. Its effective
damage-driven TED does not supply a full dynamic defect parameter set.
For C++ setup helpers, use `AnnealSetup`, `lookupAnneal`, and `applyAnnealSetup`
from `<models/psAnnealSetup.hpp>`; direct setters are available in both C++ and
Python.

## Activation and electrical output

`applyActivation(domain)` computes active concentration without running
diffusion. Configure the temperature, species label, active label, and
solubility model first. For the nm-based boron example above, the following
uses the public table's boron solubility parameters (C0 converted to nm⁻³):

```python
anneal.setActiveLabel("B_active")
anneal.enableSolidActivation(True)
anneal.setSolidSolubilityArrhenius(1e3, 0.91)
anneal.applyActivation(domain)

sheet = vps.SheetResistance()
sheet.setCellSet(domain.getCellSet())
sheet.setConcentrationLabel("B_active")
sheet.setLengthUnit(1e-7) # nm to cm; also sets concentration conversion
rsh = sheet.computeHole() # ohms per square for p-type silicon
```

`SheetResistance` uses the active concentration and a Masetti–Severi mobility
model; use `computeElectron()` for n-type silicon. If concentrations were
already converted to cm⁻³, call `setConcentrationUnit(1.0)` after
`setLengthUnit(...)`.

For multiple species, `NetDoping` combines donor and acceptor fields. Attach
the cell set, register active fields with `addDonorLabel("P_active")` and
`addAcceptorLabel("B_active")`, then call `apply()` to write `net_doping`.
`junctionDepth()` and `junctionDepths()` extract sign changes along the depth
profile; depths use the geometry length unit. Both analysis classes allow
`setDepthAxis` and `setSurfacePosition` to configure the depth coordinate.

The Python names are `vps.SheetResistance` and `vps.NetDoping`. C++ users can
access the corresponding `viennacs::SheetResistance<T, D>` and
`viennacs::NetDoping<T, D>` classes through `<csSheetResistance.hpp>` and
`<csNetDoping.hpp>`.

## Related examples

* [Ion Implantation and Anneal](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionImplantation)
* [Manual Phosphorus Implant and Anneal](https://github.com/ViennaTools/ViennaPS/tree/master/examples/pImplantManual)
* [PN Junction](https://github.com/ViennaTools/ViennaPS/tree/master/examples/pnJunction)
