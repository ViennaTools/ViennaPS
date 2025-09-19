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

An isotropic etching or deposition process applies to all materials in the domain except the mask material, which defaults to `Material::Undefined` (no mask). A negative rate etches the material, while a positive rate deposits the material of the top level set. To deposit a different material, call `duplicateTopLevelSet` on the domain before running the process.

```c++
psIsotropicProcess(const NumericType rate,
                   const Material maskMaterial = Material::Undefined)
```

| Parameter | Description | Type |
|-----------|-------------|------|
| `rate` | Rate of the process. | `NumericType` |
| `maskMaterial` | Material that does not participate in the process. | `Material` |

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

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                             20. /*yExtent*/, 10. /*trenchWidth*/,
                             10. /*trenchDepth*/, 0., 0., false, false,
                             Material::Si)
      .apply();
  // duplicate top layer to capture deposition
  domain->duplicateTopLevelSet(Material::SiO2);

  auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
      0.1 /*rate*/, Material::None);

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