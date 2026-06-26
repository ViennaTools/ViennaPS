---
layout: default
title: Thermal Oxidation
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 14
---

# Thermal Oxidation
{: .fs-9 .fw-500}

```c++
#include <psOxidation.hpp>
```
---

`Oxidation` models thermal silicon oxidation using the coupled oxidation solver
provided by ViennaLS. The model exposes process-level controls such as furnace
temperature, oxidation time, oxidant species, pressure, and crystal orientation,
while the underlying solver handles oxidant diffusion, interface motion, and
oxide deformation.

Unlike most particle- or velocity-field-based process models, `Oxidation`
manages its own physics. It is applied through the normal `Process` interface,
but the model internally performs the required coupled diffusion and deformation
steps.

## Physical Model

The built-in rate constants follow a Deal-Grove style thermal oxidation model.
The model supports dry oxidation (`O2`) and wet oxidation (`H2O`) and includes
crystal-orientation-dependent reaction-rate corrections for silicon.

During each internal time step, the solver computes:

* oxidant diffusion through the oxide,
* reaction at the Si/SiO2 interface,
* oxide volume expansion,
* deformation and pressure inside the oxide, and
* stress feedback on diffusion and reaction rates.

If a Si3N4 mask is present in the material stack, the model activates LOCOS
physics. In that mode, oxidant transport below the nitride mask, mask bending,
and constrained oxide expansion produce the characteristic bird's-beak shape.

## Domain Requirements

The domain must contain a silicon material level set, typically `Material::Si`
or `Material::BulkSi`. If no SiO2 level set is present, the model can create a
thin native oxide seed internally.

Common material roles are:

| Role | Default material |
|------|------------------|
| Silicon | `Material::Si` |
| Oxide | `Material::SiO2` |
| LOCOS mask | `Material::Si3N4` |

The material roles can be overridden with `setSiliconMaterial`,
`setOxideMaterial`, and `setMaskMaterial`.

## Main Parameters

| Method | Description |
|--------|-------------|
| `setTemperature(temperatureC)` | Furnace temperature in degrees Celsius. |
| `setTime(timeHr)` | Total oxidation time in hours. |
| `setOxidant(oxidant)` | Oxidant species: `OxidantType::Dry` or `OxidantType::Wet`. |
| `setPressure(pressureAtm)` | Ambient pressure in atm; scales both linear and parabolic rates. |
| `setOrientation(orientation)` | Silicon orientation: `Si100`, `Si110`, `Si111`, or `PolySi`. |
| `setTimeStep(dtHr)` | Optional maximum internal time step in hours. The actual step remains CFL-limited. |
| `setCFLFactor(factor)` | CFL factor for internal interface motion, clamped below 0.5. |
| `setInitialOxideThickness(thicknessUm)` | Native oxide seed thickness in micrometers if no oxide layer exists. |
| `setTransferCoefficient(coefficient)` | Gas-transfer coefficient at the oxide/ambient interface. |
| `setMaxGridPoints(maxGridPoints)` | Caps the Cartesian grid size used by the ViennaLS solves. |

## Stress and Solver Controls

| Method | Description |
|--------|-------------|
| `setReactionActivationVolume(volume)` | Stress coupling for the interface reaction rate. |
| `setDiffusionActivationVolume(volume)` | Stress coupling for oxide diffusivity. |
| `setCouplingIterations(iterations)` | Maximum outer diffusion-deformation coupling iterations. |
| `setCouplingTolerance(tolerance)` | Convergence tolerance for the outer coupling loop. |
| `setMechanicsIterations(iterations)` | Maximum oxide mechanics iterations. |
| `setPressureIterations(iterations)` | Maximum pressure-solve iterations. |
| `setStokesIterations(iterations)` | Maximum Stokes velocity iterations. |
| `setPressureTolerance(tolerance)` | Pressure-solve tolerance. |
| `setStokesTolerance(tolerance)` | Stokes-solve tolerance. |
| `setMechanicsTolerance(tolerance)` | Mechanics coupling tolerance. |
| `setSimpleVelocityRelaxation(alpha)` | SIMPLE velocity relaxation factor. |
| `setSimplePressureRelaxation(beta)` | SIMPLE pressure relaxation factor. |
| `setSolveBounds(minIndex, maxIndex)` | Restricts the solve to explicit Cartesian index bounds. |
| `clearSolveBounds()` | Clears explicit solve bounds. |

## LOCOS Mask Controls

These controls are used when the domain contains the configured mask material,
by default `Material::Si3N4`.

| Method | Description |
|--------|-------------|
| `setMaskMaterial(material)` | Selects the material treated as the oxidation mask. |
| `setMaskParameters(params)` | Replaces the full ViennaLS mask parameter object. |
| `setMaskTractionIterations(iterations)` | Maximum mask-traction iterations. |
| `setMaskTractionTolerance(tolerance)` | Mask-traction convergence tolerance. |
| `setMaskTractionRelaxation(relaxation)` | Outer mask-traction relaxation factor. |
| `setMaskContactLoadRelaxation(relaxation)` | Contact-load relaxation factor. |
| `setMaskContactReleaseFraction(fraction)` | Fraction used for contact release. |
| `setMaskUnilateralContact(enabled)` | Enables or disables unilateral contact. |
| `setMaskSmootherOmega(omega)` | Multigrid smoother over-relaxation for mask mechanics. |
| `setMaskBendingBounds(minIndex, maxIndex)` | Restricts mask bending to explicit bounds. |
| `clearMaskBendingBounds()` | Clears explicit mask bending bounds. |
| `setMaskCouplingIterations(iterations)` | Maximum mask/deformation coupling iterations. |
| `setMaskCouplingTolerance(tolerance)` | Mask/deformation coupling tolerance. |

## GPU Controls

The default solver mode is CPU. GPU acceleration is available when ViennaLS was
built with GPU support.

| Method | Description |
|--------|-------------|
| `setGpuMode(mode)` | Selects `GpuMode::Cpu` or `GpuMode::Gpu`. |
| `setGpuPreconditioner(preconditioner)` | Selects the GPU BiCGSTAB preconditioner, such as Jacobi or ILU0. |

## Output Helpers

| Method | Description |
|--------|-------------|
| `estimatePlanarOxideThickness(initialOxideThickness)` | Returns the planar Deal-Grove oxide thickness estimate for the current settings. |
| `saveSurfaceMesh(domain, fileName)` | Writes a wrapped surface mesh without mutating the active simulation state. |
| `saveVolumeMesh(domain, baseName)` | Writes a wrapped volume mesh and associated oxide-field output. |

## Example Usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
using namespace viennaps;

auto model = SmartPointer<Oxidation<double, 2>>::New();
model->setTemperature(1000.0);
model->setTime(0.2);
model->setOxidant(OxidantType::Wet);
model->setPressure(1.0);
model->setOrientation(SiliconOrientation::Si100);
model->setMaxGridPoints(5000000);

Process<double, 2>(domain, model).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
import viennaps as vps

model = vps.Oxidation()
model.setTemperature(1000.0)
model.setTime(0.2)
model.setOxidant(vps.OxidantType.Wet)
model.setPressure(1.0)
model.setOrientation(vps.SiliconOrientation.Si100)
model.setMaxGridPoints(5_000_000)

vps.Process(domain, model).apply()
```
</details>

## Practical Notes

* Use smaller `gridDelta` values for production results, but keep in mind that
  the diffusion/deformation grid scales strongly with dimension.
* `setTimeStep` is an upper bound. The accepted internal step is still limited
  by the CFL condition.
* LOCOS simulations need enough vertical space above the mask and oxide so that
  the growing oxide and bent mask are not clipped by the domain bounds.
* In narrow or confined geometries, stress feedback can strongly reduce the
  local reaction rate. Inspect pressure and stress fields in the oxide-field
  output to understand growth suppression.

## Related Examples

* [Step Oxidation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/stepOxidation)
* [Fin Oxidation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/finOxidation)
* [Trench Oxidation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchOxidation)
* [LOCOS Oxidation](https://github.com/ViennaTools/ViennaPS/tree/master/examples/locosOxidation)
