---
layout: default
title: Isotropic Process
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 1
---

# Isotropic Process
{: .fs-9 .fw-500}

```c++
#include <psIsotropicProcess.hpp>
```
---

An isotropic etching or deposition process moves the surface with a material-dependent normal velocity. A negative rate etches the material, while a positive rate deposits the material of the top level set. To deposit a different material, call `duplicateTopLevelSet` on the domain before running the process.

The simplest constructor applies one default rate to all materials and optionally masks one or more materials by assigning them a rate of `0`. For selective processes, material-specific rates can be passed directly with a default fallback rate for all other materials.

```c++
IsotropicProcess(NumericType isotropicRate,
                 Material maskMaterial = Material::Undefined)

IsotropicProcess(NumericType isotropicRate,
                 const std::vector<Material> &maskMaterials)

IsotropicProcess(std::unordered_map<Material, NumericType> materialRates,
                 NumericType defaultRate = 0.)
```

| Parameter | Description | Type |
|-----------|-------------|------|
| `isotropicRate` | Default rate of the process. | `NumericType` |
| `maskMaterial` | Material that does not participate in the process. Defaults to `Material::Undefined`. | `Material` |
| `maskMaterials` | Materials that do not participate in the process. | `std::vector<Material>` |
| `materialRates` | Material-specific process rates. | `std::unordered_map<Material, NumericType>` |
| `defaultRate` | Fallback rate for materials not listed in `materialRates`. | `NumericType` |

Material rates can also be changed after construction:

```c++
void setIsotropicRate(NumericType isotropicRate)
void setMaterialRate(Material material, NumericType rate)
```

| Method | Description |
|--------|-------------|
| `setIsotropicRate` | Updates the default isotropic rate used for materials without a specific rate. |
| `setMaterialRate` | Sets or updates the rate for one material. Use a rate of `0` to mask that material. |

__Material-specific example:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
std::unordered_map<Material, NumericType> rates = {
    {Material::Si, -1.0},
    {Material::SiO2, -0.2},
    {Material::Mask, 0.0},
};

auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
    rates, 0.0 /*defaultRate*/);
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
rates = {
    vps.Material.Si: -1.0,
    vps.Material.SiO2: -0.2,
    vps.Material.Mask: 0.0,
}

model = vps.IsotropicProcess(materialRates=rates, defaultRate=0.0)
```
</details>

__Deposition example:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
#include <psIsotropicProcess.hpp>
#include <psMakeTrench.hpp>
#include <psProcess.hpp>

using namespace viennaps;

int main() {
  using NumericType = double;
  constexpr int D = 2;

  auto domain = Domain<NumericType, D>::New();
  MakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                             20. /*yExtent*/, 10. /*trenchWidth*/,
                             10. /*trenchDepth*/, 0., 0., false, false,
                             Material::Si)
      .apply();
  // duplicate top layer to capture deposition
  domain->duplicateTopLevelSet(Material::SiO2);

  auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
      0.1 /*rate*/);

  domain->saveVolumeMesh("trench_initial");
  Process<NumericType, D>(domain, model, 20.).apply(); // run process for 20s
  domain->saveVolumeMesh("trench_final");
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

domain = vps.Domain()
vps.MakeTrench(domain=domain,
               gridDelta=0.1,
               xExtent=20.0,
               yExtent=20.0,
               trenchWidth=10.0,
               trenchDepth=10.0,
               taperingAngle=0.0,
               baseHeight=0.0,
               periodicBoundary=False,
               makeMask=False,
               material=vps.Material.Si
              ).apply()
# duplicate top layer to capture deposition
domain.duplicateTopLevelSet(vps.Material.SiO2)

model = vps.IsotropicProcess(rate=0.1)

domain.saveVolumeMesh("trench_initial")
vps.Process(domain, model, 20.0).apply()
domain.saveVolumeMesh("trench_final")
```
</details>

Results:
![]({% link assets/images/isotropicDeposition.png %})

__Etching example:__
<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
#include <psIsotropicProcess.hpp>
#include <psMakeTrench.hpp>
#include <psProcess.hpp>

using namespace viennaps;

int main() {
  using NumericType = double;
  constexpr int D = 2;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                             20. /*yExtent*/, 5. /*trenchWidth*/,
                             5. /*trenchDepth*/, 0., 0., false, true /*makeMask*/,
                             Material::Si)
      .apply();

  auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
      -0.1 /*rate*/, Material::Mask);

  domain->saveVolumeMesh("trench_initial");
  Process<NumericType, D>(domain, model, 50.).apply(); // run process for 20s
  domain->saveVolumeMesh("trench_final");
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

domain = vps.Domain()
vps.MakeTrench(domain=domain,
               gridDelta=0.1,
               xExtent=20.0,
               yExtent=20.0,
               trenchWidth=5.0,
               trenchDepth=5.0,
               taperingAngle=0.0,
               baseHeight=0.0,
               periodicBoundary=False,
               makeMask=True,
               material=vps.Material.Si
              ).apply()

model = vps.IsotropicProcess(rate=-0.1, maskMaterial=vps.Material.Mask)

domain.saveVolumeMesh("trench_initial", True)
vps.Process(domain, model, 50.0).apply()
domain.saveVolumeMesh("trench_final", True)
```
</details>
