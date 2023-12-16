---
layout: default
title: Trench Geometry
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 2
---

# Trench Geometry
{: .fs-9 .fw-500 }

```c++
#include <psMakeTrench.hpp> 
```
---

The `psMakeTrench` class is used to generate a new trench geometry extending in the z (3D) or y (2D) direction, centrally
positioned at the origin with the total extent specified in the x and y directions. The
trench configuration may include periodic boundaries in both the x and y directions.
Users have the flexibility to define the trench’s width, depth, and incorporate tapering
with a designated angle. Moreover, the trench can serve as a mask, applying the
specified material exclusively to the bottom while the remaining portion adopts the
mask material.

```c++
psMakeTrench(psDomainType domain, 
             const NumericType gridDelta,
             const NumericType xExtent, 
             const NumericType yExtent,
             const NumericType trenchWidth,
             const NumericType trenchDepth,
             const NumericType taperingAngle = 0., // in degrees
             const NumericType baseHeight = 0.,
             const bool periodicBoundary = false,
             const bool makeMask = false,
             const psMaterial material = psMaterial::None)
```

| Parameter              | Description                                                         | Type                           |
|------------------------|---------------------------------------|--------------------------------|
| `domain`               | Specifies the type of domain for the trench geometry.                   |   `psSmartPointer<psDomain<NumericType, D>>` |
| `gridDelta`            | Represents the grid spacing or resolution used in the simulation.                                          | `NumericType`    |
| `xExtent`              | Defines the extent of the trench geometry in the x-direction.                                               | `NumericType`    |
| `yExtent`              | Specifies the extent of the trench geometry in the y-direction.                                             | `NumericType`    |
| `trenchWidth`          | Sets the width of the trench.                                                                             | `NumericType`    |
| `trenchDepth`          | Determines the depth of the trench.                                                                       | `NumericType`    |
| `taperingAngle`        | (Optional) Specifies the angle of tapering for the trench geometry in degrees. Default is set to 0.         | `NumericType`    |
| `baseHeight`           | (Optional) Sets the base height of the trench. Default is set to 0.                                         | `NumericType`    |
| `periodicBoundary`     | (Optional) If set to true, enables periodic boundaries in both x and y directions (only applicable in 3D). Default is set to false. | `bool`   |
| `makeMask`             | (Optional) If set to true, allows the trench to function as a mask, with specified material applied only to the bottom. Default is set to false. | `bool`                  |
| `material`             | (Optional) Specifies the material used for the trench. Default is set to `psMaterial_None`.                |    `psMaterial`               |

__Example usage__:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
psMakeTrench<NumericType, D>(domain, 0.5, 10.0, 10.0, 5.0, 5.0, 10., 0.,
                            false, false, psMaterial::Si)
    .apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
domain = vps.Domain()
vps.MakeTrench(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              trenchWidth=5.0,
              trenchDepth=5.0,
              taperingAngle=10.0,
              baseHeight=0.0,
              periodicBoundary=False,
              makeMask=False,
              material=vps.Material.Si,
             ).apply()
```
</details>