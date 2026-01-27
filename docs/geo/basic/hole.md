---
layout: default
title: Hole Geometry
parent: Geometry Builders
grand_parent: Creating a Geometry
nav_order: 3
---

# Hole Geometry
{: .fs-9 .fw-500 }

```c++
#include <geometries/psMakeHole.hpp> 
```
---

The `MakeHole` class generates a hole geometry in the z direction, which, in 2D mode, corresponds to a trench geometry. Positioned at the origin, the hole is centered, with the total extent defined in the x and y directions. The normal direction for the hole creation is in the positive z direction in 3D and the positive y direction in 2D. Users can specify the hole's radius, depth, and opt for tapering with a designated angle. The hole configuration may include periodic boundaries in both the x and y directions. 
Additionally, the hole can serve as a mask, with the specified material only applied to the bottom of the hole, while the remainder adopts the mask material.

```c++
// namespace viennaps

// with DomainSetup configured (v3.3.0)
MakeHole(pviennaps::Domain domain, 
         NumericType holeRadius, 
         NumericType holeDepth,
         NumericType holeTaperAngle = 0., 
         NumericType maskHeight = 0.,
         NumericType maskTaperAngle = 0., 
         HoleShape shape = HoleShape::Full,
         Material material = Material::Si,
         Material maskMaterial = Material::Mask)

MakeHole(viennaps::Domain domain,
         NumericType gridDelta,
         NumericType xExtent, 
         NumericType yExtent,
         NumericType holeRadius,
         NumericType holeDepth,
         NumericType taperAngle = 0., // in degrees
         NumericType baseHeight = 0.,
         bool periodicBoundary = false,
         bool makeMask = false,
         Material material = Material::Si,
         HoleShape holeShape = HoleShape::Full)
```

| Parameter          | Type           | Description  | Applicable Constructor |
|-------------------|---------------|--------------|------------------------|
| `domain`         | `psDomainType` | The simulation domain. | Both |
| `holeRadius`     | `NumericType`  | Radius of the hole. | Both |
| `holeDepth`      | `NumericType`  | Depth of the hole. | Both |
| `holeTaperAngle` | `NumericType`  | Taper angle of the hole (default: `0.`). | Both |
| `maskHeight`     | `NumericType`  | Height of the masking layer (default: `0.`). | First constructor only |
| `maskTaperAngle` | `NumericType`  | Taper angle of the masking layer (default: `0.`). | First constructor only |
| `shape`          | `HoleShape`    | Shape of the hole (default: `HoleShape::Full`). | Both |
| `material`       | `Material`     | Material of the hole (default: `Material::Si`). | Both |
| `maskMaterial`   | `Material`     | Material of the mask (default: `Material::Mask`). | First constructor only |
| `gridDelta`      | `NumericType`  | Grid spacing in the simulation domain. | Second constructor only |
| `xExtent`        | `NumericType`  | Extent of the domain in the x-direction. | Second constructor only |
| `yExtent`        | `NumericType`  | Extent of the domain in the y-direction. | Second constructor only |
| `taperAngle`     | `NumericType`  | Alternative name for `holeTaperAngle` in the second constructor (default: `0.`). | Second constructor only |
| `baseHeight`     | `NumericType`  | Height at which the hole starts (default: `0.`). | Second constructor only |
| `periodicBoundary` | `bool`       | If `true`, enables periodic boundary conditions (default: `false`). | Second constructor only |
| `makeMask`       | `bool`         | If `true`, the mask is created instead of the hole, setting `holeDepth` to `0` (default: `false`). | Second constructor only |

> **Note**:  
> - The first constructor requires that the domain is already configured.  
> - The second constructor sets up a new simulation domain with `gridDelta`, `xExtent`, and `yExtent`.  
> - If `makeMask` is `true`, the hole is not created, and only the masking layer is applied.  

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps

// with DomainSetup configured (v3.3.0)
auto domain = Domain<NumericType, D>::New(0.5, 10., 10., BoundaryType::REFLECTIVE_BOUNDARY);
MakeHole<NumericType, D>(domain, 5.0, 5.0, 10., 0., 0., HoleShape::Quarter, Material::Si, Material::Mask)
    .apply();

// without DomainSetup
auto domain = Domain<NumericType, D>::New();
MakeHole<NumericType, D>(domain, 0.5, 10.0, 10.0, 2.5, 5.0, 10., 0., false,
                         false, Material::Si)
    .apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
# with DomainSetup configured (v3.3.0)
domain = vps.Domain(gridDelta=0.5, 
                    xExtent=10.0, 
                    yExtent=10.0, 
                    boundaryType=vps.BoundaryType.REFLECTIVE_BOUNDARY)
vps.MakeHole(domain=domain,
             holeRadius=5.0,
             holeDepth=0.0,
             holeTaperAngle=0.0,
             maskHeight=5.0,
             maskTaperAngle=2.0,
             shape=vps.HoleShape.Quarter,
             material=vps.Material.Si,
             maskMaterial=vps.Material.Mask
            ).apply()

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
              holeShape=vps.HoleShape.Quarter
             ).apply()
```
</details>