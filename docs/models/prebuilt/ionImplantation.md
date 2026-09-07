---
layout: default
title: Ion Implantation
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 16
---

# Ion Implantation
{: .fs-9 .fw-500}

```c++
#include <models/psIonImplantation.hpp>
```

`IonImplantation` deposits an analytical dopant profile and optional damage
profile into the domain's [cell set]({% link domain/volume.md %}). It supports
beam tilt, masking, screen materials, and dose normalization. Apply it through
`Process` with duration `0.0`; the dose is configured on the implant model.
The level-set geometry is unchanged.

## Profiles and materials

| Profile | Purpose |
|---------|---------|
| `ImplantPearsonIV` | A Pearson IV depth profile with Gaussian lateral spread. |
| `ImplantDualPearsonIV` | Weighted head and tail profiles with separate lateral spreads. |
| `ImplantPearsonIVChanneling` | A Pearson IV profile with an exponential channeling tail. |
| `ImplantDamageHobler` | Optional damage depth and lateral distribution. |
| `ImplantRecipeModel`, `DamageRecipeModel` | Profiles constructed from explicit recipe entries or CSV table lookup. |

`setMaskMaterials(...)` selects materials that block the beam.
`setScreenMaterials(...)` selects materials the beam traverses before reaching
the implant surface. Marking a layer as a screen does not itself configure an
energy-loss model; choose profile parameters or a recipe appropriate to the
screen thickness.

## Example: implant a silicon substrate

This example uses illustrative profile parameters and a geometry measured in
nanometers. Two silicon planes bound the substrate region used for the cell
set. The air cover extends to 20 nm above the surface.

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```c++
#include <geometries/psMakePlane.hpp>
#include <models/psIonImplantation.hpp>
#include <process/psProcess.hpp>

namespace ps = viennaps;

int main() {
  using T = double;
  constexpr int D = 2;
  auto domain = ps::Domain<T, D>::New(
      2.0, 100.0, ps::BoundaryType::REFLECTIVE_BOUNDARY);
  ps::MakePlane<T, D>(domain, -200.0, ps::Material::Si).apply();
  ps::MakePlane<T, D>(domain, 0.0, ps::Material::Si, true).apply();
  domain->generateCellSet(20.0, ps::Material::Air, true, true);
  domain->getCellSet()->buildNeighborhood();

  ps::PearsonIVParameters<T> moments;
  moments.mu = 60.0;
  moments.sigma = 20.0;
  moments.gamma = 0.5;
  moments.beta = 4.0;
  auto profile = ps::SmartPointer<ps::ImplantPearsonIV<T, D>>::New(
      moments, 0.0, 25.0);

  auto implant = ps::SmartPointer<ps::IonImplantation<T, D>>::New();
  implant->setImplantModel(profile);
  implant->setDose(1e13);
  implant->setTiltAngle(7.0);
  implant->setLengthUnit(1e-7);
  implant->setDoseControl(ps::ImplantDoseControl::WaferDose);
  implant->setMaskMaterials({ps::Material::Mask});
  implant->setScreenMaterials({ps::Material::SiO2});
  implant->setConcentrationLabel("B_total");
  ps::Process<T, D>(domain, implant, 0.0).apply();
  domain->getCellSet()->writeVTU("post_implant");
}
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>

```python
import viennaps as vps

vps.setDimension(2)
domain = vps.Domain(2.0, 100.0)
vps.MakePlane(domain, -200.0, vps.Material.Si).apply()
vps.MakePlane(domain, 0.0, vps.Material.Si, True).apply()
domain.generateCellSet(20.0, vps.Material.Air, True, True)
domain.getCellSet().buildNeighborhood()

moments = vps.PearsonIVParameters()
moments.mu = 60.0
moments.sigma = 20.0
moments.gamma = 0.5
moments.beta = 4.0
profile = vps.ImplantPearsonIV(moments, 0.0, 25.0)

implant = vps.IonImplantation()
implant.setImplantModel(profile)
implant.setDose(1e13)
implant.setTiltAngle(7.0)
implant.setLengthUnit(1e-7)
implant.setDoseControl(vps.ImplantDoseControl.WaferDose)
implant.setMaskMaterials([vps.Material.Mask])
implant.setScreenMaterials([vps.Material.SiO2])
implant.setConcentrationLabel("B_total")
vps.Process(domain, implant, 0.0).apply()
domain.getCellSet().writeVTU("post_implant")
```
</details>

## Units and output

Dose is in ions/cm², tilt is in degrees, and `setLengthUnit` takes centimeters
per geometry unit (`1e-7` for nm). `WaferDose` normalizes dose on the wafer
plane; `BeamDose` uses the beam-normal dose. `Off` disables dose control.

Concentrations are stored in geometry unit⁻³ by default (nm⁻³ in the example).
`setOutputConcentrationInCm3(true)` converts implant output to cm⁻³. Keep the
native units when passing fields to an anneal configured in geometry units.
Use `setDamageLabel`, `setLastDamageLabel`, and `setBeamHitsLabel` to name
additional fields; `enableBeamHits()` enables beam-hit diagnostics.

The `withEmbeddedBoundaries` option stores sub-grid boundary information for
tilted implants. Set it when generating the cell set; enabling it later on the
implant can rebuild the cell set.

For table-based setup and available data, see the
[Model Database]({% link models/modelDatabase.md %}). To diffuse or activate the
resulting profile, continue with [Thermal Annealing]({% link models/prebuilt/anneal.md %}).

## Related examples

* [Ion Implantation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionImplantation)
* [Manual Phosphorus Implant](https://github.com/ViennaTools/ViennaPS/tree/master/examples/pImplantManual)
* [PN Junction](https://github.com/ViennaTools/ViennaPS/tree/master/examples/pnJunction)
