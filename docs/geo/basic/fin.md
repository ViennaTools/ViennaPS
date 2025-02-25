---
layout: default
title: Fin Geometry
parent: Geometry Builders
grand_parent: Creating a Geometry
nav_order: 4
---

# Fin Geometry
{: .fs-9 .fw-500 }

```c++
#include <geometries/psMakeFin.hpp>
```
---

The `MakeFin` class generates a fin geometry extending in the z (3D) or y (2D) direction, centered at the origin with specified dimensions in the x and y directions. The fin may incorporate periodic boundaries in the x and y directions (limited to 3D). Users can define the width and height of the fin, and it can function as a mask, with the specified material exclusively applied to the bottom of the fin, while the upper portion adopts the mask material.

```c++
// namespace viennaps

// with DomainSetup configured (v3.3.0)
MakeFin(DomainType domain,
        NumericType finWidth,
        NumericType finHeight,
        NumericType finTaperAngle,
        NumericType maskHeight = 0.,
        NumericType maskTaperAngle = 0.,
        bool halfFin = false,
        Material material = Material::Si,
        Material maskMaterial = Material::Mask)

MakeFin(DomainType domain,
        const NumericType gridDelta,
        const NumericType xExtent, 
        const NumericType yExtent,
        const NumericType finWidth,
        const NumericType finHeight,
        const NumericType baseHeight = 0.,
        const bool periodicBoundary = false,
        const bool makeMask = false,
        const Material material = Material::Si)
```

| Parameter           | Type           | Description  | Applicable Constructor |
|---------------------|---------------|--------------|------------------------|
| `domain`           | `psDomainType` | The simulation domain. | Both |
| `finWidth`         | `NumericType`  | Width of the fin. | Both |
| `finHeight`        | `NumericType`  | Height of the fin. | Both (ignored if `makeMask = true`) |
| `finTaperAngle`    | `NumericType`  | Taper angle of the fin (default: `0.`). | Both (ignored if `makeMask = true`) |
| `maskHeight`       | `NumericType`  | Height of the mask (default: `0.`). | First constructor only |
| `maskTaperAngle`   | `NumericType`  | Taper angle of the mask (default: `0.`). | First constructor only |
| `halfFin`          | `bool`         | If `true`, the fin is halved along the x-axis. | First constructor only |
| `material`         | `Material`     | Material of the fin (default: `Material::Si`). | Both |
| `maskMaterial`     | `Material`     | Material of the mask (default: `Material::Mask`). | First constructor only |
| `gridDelta`        | `NumericType`  | Grid spacing in the simulation domain. | Second constructor only |
| `xExtent`         | `NumericType`  | Extent of the domain in the x-direction. | Second constructor only |
| `yExtent`         | `NumericType`  | Extent of the domain in the y-direction. | Second constructor only |
| `taperAngle`      | `NumericType`  | Taper angle of the fin/mask. | Second constructor only |
| `baseHeight`      | `NumericType`  | Base height of the fin (default: `0.`). | Second constructor only |
| `periodicBoundary` | `bool`         | If `true`, enables periodic boundary conditions (default: `false`). | Second constructor only |
| `makeMask`        | `bool`         | If `true`, a mask is created instead of a fin (default: `false`). | Second constructor only |

> **Note**:  
> - The first constructor **requires** that the domain is already configured and is only available from **ViennaPS v3.3.0**.  
> - The second constructor allows domain setup within the constructor by specifying `gridDelta`, `xExtent`, and `yExtent`.  

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps
// recomended with DomainSetup configured (v3.3.0)
auto domain = SmartPointer<Domain<NumericType, D>>::New(0.5, 10., 10., BoundaryType::REFLECTIVE_BOUNDARY);
MakeFin(domain, 2.5, 5.0, 10., 0., 0., false, Material::Si, Material::Mask).apply();

// without DomainSetup
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
# with DomainSetup configured (v3.3.0)
domain = vps.Domain(0.5, 10., 10., vps.BoundaryType.REFLECTIVE_BOUNDARY)
vps.MakeFin(domain=domain,
            finWidth=2.5,
            finHeight=5.0,
            finTaperAngle=10.,
            maskHeight=5.,
            maskTaperAngle=0.,
            halfFin=False,
            material=vps.Material.Si,
            maskMaterial=vps.Material.Mask
           ).apply()

# without DomainSetup
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