---
layout: default
title: Stack Geometry
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 5
---

# Stack Geometry
{: .fs-9 .fw-500 }

```c++
#include <psMakeStack.hpp>
```
---

The `psMakeStack` generates a stack of alternating SiO<sub>2</sub>/Si<sub>3</sub>N<sub>4</sub> layers featuring an optionally etched hole (3D) or trench (2D) at the center. The stack emerges in the positive z direction (3D) or y direction (2D) and is centered around the origin, with its x/y extent specified. Users have the flexibility to introduce periodic boundaries in the x and y directions. Additionally, the stack can incorporate a top mask with a central hole of a specified radius or a trench with a designated width. This versatile functionality enables users to create diverse and customized structures for simulation scenarios.

```c++
psMakeStack(psDomainType domain, 
            const NumericType gridDelta,
            const NumericType xExtent, 
            const NumericType yExtent,
            const int numLayers, 
            const NumericType layerHeight,
            const NumericType substrateHeight,
            const NumericType holeRadius,
            const NumericType trenchWidth,
            const NumericType maskHeight, 
            const bool periodicBoundary = false)
```

| Parameter              | Description                                                       | Type                           |
|------------------------|-------------------------------------------------------------------|--------------------------------|
| `domain`               | Specifies the domain type for the stack geometry.                   | `psSmartPointer<psDomain<NumericType, D>>` |
| `gridDelta`            | Represents the grid spacing or resolution used in the simulation.             | `NumericType` |
| `xExtent`              | Defines the extent of the stack geometry in the x-direction.                   | `NumericType` |
| `yExtent`              | Specifies the extent of the stack geometry in the y-direction.                 | `NumericType` |
| `numLayers`            | Sets the number of layers in the stack.                          | `int` |
| `layerHeight`          | Determines the height of each layer in the stack.                      | `NumericType` |
| `substrateHeight`      | Specifies the height of the substrate.                                 | `NumericType` |
| `holeRadius`           | Sets the radius of the hole.                                             | `NumericType` |
| `trenchWidth`          | Determines the width of the trench.                                         | `NumericType` |
| `maskHeight`           | Specifies the height of the mask.                                             | `NumericType` |
| `periodicBoundary`     | (Optional) If set to true, enables periodic boundaries. Default is set to false.        | `bool` |

{: .note}
> `trenchWidth` and `holeRadius` can only be used mutually exclusive. I.e., if one is set, the other has to be set to 0.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
psMakeStack<NumericType, D>(domain, 0.5, 10.0, 10.0, 5, 5.0, 10., 0.0, 5.0,
                            0.0, false)
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
vps.MakeStack(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              numLayers=5,
              layerHeight=5.0,
              substrateHeight=10.0,
              holeRadius=0.0,
              trenchWidth=5.0,
              maskHeight=0.0,
              periodicBoundary=False,
             ).apply()
```
</details>