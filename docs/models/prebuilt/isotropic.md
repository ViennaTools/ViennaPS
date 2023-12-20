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

An isotropic etching or deposition process initiates across all materials in the domain, excluding the masking material, which is by default set to `psMaterial::None`. The default setting means, that the process unfolds uniformly across all materials within the domain. When the rate is less than 0, the material undergoes etching. Conversely, when the rate exceeds 0, material deposition occurs in accordance with the material of the top level set. If you want to deposit a new material, make sure to call the function `duplicateTopLevelSet` in your domain instance.

```c++
psIsotropicProcess(const NumericType rate,
                   const psMaterial maskMaterial = psMaterial::None)
```

| Parameter | Description | Type |
|-----------|-------------|------|
| `rate` | Rate of the process. | `NumericType` |
| `maskMaterial` | Material that does not participate in the process. | `psMaterial` |

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

int main() {
  using NumericType = double;
  constexpr int D = 2;

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                               20. /*yExtent*/, 10. /*trenchWidth*/,
                               10. /*trenchDepth*/, 0., 0., false, false,
                               psMaterial::Si)
      .apply();
  // duplicate top layer to capture deposition
  domain->duplicateTopLevelSet(psMaterial::SiO2);

  auto model = psSmartPointer<psIsotropicProcess<NumericType, D>>::New(
      0.1 /*rate*/, psMaterial::None);

  domain->saveVolumeMesh("trench_initial");
  psProcess<NumericType, D>(domain, model, 20.).apply(); // run process for 20s
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
import viennaps2d as vps

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
![](../../assets/images/isotropicDeposition.png)

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

int main() {
  using NumericType = double;
  constexpr int D = 2;

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                               20. /*yExtent*/, 5. /*trenchWidth*/,
                               5. /*trenchDepth*/, 0., 0., false, true /*makeMask*/,
                               psMaterial::Si)
      .apply();

  auto model = psSmartPointer<psIsotropicProcess<NumericType, D>>::New(
      -0.1 /*rate*/, psMaterial::Mask);

  domain->saveVolumeMesh("trench_initial");
  psProcess<NumericType, D>(domain, model, 50.).apply(); // run process for 20s
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
import viennaps2d as vps

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