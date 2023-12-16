---
layout: default
title: Hole Geometry
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 3
---

# Hole Geometry
{: .fs-9 .fw-500 }

```c++
#include <psMakeHole.hpp> 
```
---

The `psMakeHole` class generates a hole geometry in the z direction, which, in 2D mode, corresponds to a trench geometry. Positioned at the origin, the hole is centered, with the total extent defined in the x and y directions. The normal direction for the hole creation is in the positive z direction in 3D and the positive y direction in 2D. Users can specify the hole's radius, depth, and opt for tapering with a designated angle. The hole configuration may include periodic boundaries in both the x and y directions. 
Additionally, the hole can serve as a mask, with the specified material only applied to the bottom of the hole, while the remainder adopts the mask material.

```c++
psMakeHole(psDomainType domain,
           const NumericType gridDelta,
           const NumericType xExtent, 
           const NumericType yExtent,
           const NumericType holeRadius,
           const NumericType holeDepth,
           const NumericType taperAngle = 0., // in degrees
           const NumericType baseHeight = 0.,
           const bool periodicBoundary = false,
           const bool makeMask = false,
           const psMaterial material = psMaterial::None)
```

| Parameter              | Description                                                           | Type                           |
|------------------------|-----------------------------------------------------------------------|--------------------------------|
| `domain`               | Specifies the type of domain for the hole geometry.                |  `psSmartPointer<psDomain<NumericType, D>>`  |
| `gridDelta`            | Represents the grid spacing or resolution used in the simulation.                   | `NumericType`                  |
| `xExtent`              | Defines the extent of the hole geometry in the x-direction.                         | `NumericType`                  |
| `yExtent`              | Specifies the extent of the hole geometry in the y-direction.                  | `NumericType`                  |
| `holeRadius`           | Sets the radius of the hole.                                   | `NumericType`                  |
| `holeDepth`            | Determines the depth of the hole.                                       | `NumericType`                  |
| `taperAngle`           | (Optional) Specifies the angle of tapering for the hole geometry in degrees. Default is set to 0. | `NumericType`      |
| `baseHeight`           | (Optional) Sets the base height of the hole. Default is set to 0.             | `NumericType`                  |
| `periodicBoundary`     | (Optional) If set to true, enables periodic boundaries in both x and y directions. Default is set to false. | `bool`  |
| `makeMask`             | (Optional) If set to true, allows the hole to function as a mask, with specified material applied only to the bottom. Default is set to false. | `bool`  |
| `material`             | (Optional) Specifies the material used for the hole. Default is set to `psMaterial::None`.    |   `psMaterial`  |

__Example usage:__

C++:
{: .label .label-blue }
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
psMakeHole<NumericType, D>(domain, 0.5, 10.0, 10.0, 2.5, 5.0, 10., 0., false,
                            false, psMaterial::Si)
    .apply();
```

Python:
{: .label .label-green }
```python
domain = vps.Domain()
vps.MakeHole(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              holeRadius=2.5,
              holeDepth=5.0,
              taperingAngle=10.0,
              baseHeight=0.0,
              periodicBoundary=False,
              makeMask=False,
              material=vps.Material.Si,
             ).apply()
```