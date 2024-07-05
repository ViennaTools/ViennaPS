---
layout: default
title: Fin Geometry
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 4
---

# Fin Geometry
{: .fs-9 .fw-500 }

```c++
#include <psMakeFin.hpp>
```
---

The `MakeFin` class generates a fin geometry extending in the z (3D) or y (2D) direction, centered at the origin with specified dimensions in the x and y directions. The fin may incorporate periodic boundaries in the x and y directions (limited to 3D). Users can define the width and height of the fin, and it can function as a mask, with the specified material exclusively applied to the bottom of the fin, while the upper portion adopts the mask material.

```c++
// namespace viennaps
MakeFin(DomainType domain,
        const NumericType gridDelta,
        const NumericType xExtent, 
        const NumericType yExtent,
        const NumericType finWidth,
        const NumericType finHeight,
        const NumericType baseHeight = 0.,
        const bool periodicBoundary = false,
        const bool makeMask = false,
        const Material material = Material::None)
```

| Parameter              | Description                                                         | Type                           |
|------------------------|---------------------------------------------------------------------|--------------------------------|
| `domain`               | Specifies the type of domain for the fin geometry.                 |  `SmartPointer<Domain<NumericType, D>` |
| `gridDelta`            | Represents the grid spacing or resolution used in the simulation.          | `NumericType`  |
| `xExtent`              | Defines the extent of the fin geometry in the x-direction.                           | `NumericType`  |
| `yExtent`              | Specifies the extent of the fin geometry in the y-direction.                            | `NumericType`  |
| `finWidth`             | Sets the width of the fin.                                              | `NumericType`  |
| `finHeight`            | Determines the height of the fin.                                          | `NumericType`  |
| `baseHeight`           | (Optional) Sets the base height of the fin. Default is set to 0.              | `NumericType`  |
| `periodicBoundary`     | (Optional) If set to true, enables periodic boundaries in both x and y directions. Default is set to false. | `bool` |
| `makeMask`             | (Optional) If set to true, allows the fin to function as a mask, with specified material applied only to the bottom. Default is set to false. | `bool`    |
| `material`             | (Optional) Specifies the material used for the fin. Default is set to `Material::None`.   |   `Material`   |

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps
auto domain = SmartPointer<Domain<NumericType, D>>::New();
MakeFin<NumericType, D>(domain, 0.5, 10.0, 10.0, 5.0, 5.0, 0., false, false,
                        Material::Si)
    .apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python:
{: .label .label-green }
</summary>
```python
domain = vps.Domain()
vps.MakeFin(domain=domain,
            gridDelta=0.5,
            xExtent=10.0,
            yExtent=10.0,
            finWidth=2.5,
            finHeight=5.0,
            baseHeight=0.0,
            periodicBoundary=False,
            makeMask=False,
            material=vps.Material.Si,
           ).apply()
```
</details>