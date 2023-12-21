---
layout: default
title: Volume
parent: Simulation Domain
nav_order: 1
---

# Volume
{: .fs-9 .fw-500 }

---

A Cell-Set (CS) is used to store and track volume information in the domain.
The CS is stored over the whole simulation domain, above and/or below the surface described by the Level-Set. It uses the same grid as the Level-Set, such that each Level-Set grid point acts as the corner of the cells around the point.
To determine which material region a cell lies in, the LS values at the cell corner of the material interface are inspected. If the sum of the values is negative the cell belongs to the underlying material, otherwise it represents the material on the other side of the interface.

To find a cell in which an arbitrary point in space resides, a bounding volume hierarchy (BVH) is built on top of the CS. This allows for fast access to random cells in the CS. Additionally, cell-neighbor relations are established when setting up the CS. This allows for quick access to neighbor cells.


## How to use the Cell-Set
{: .lh-default}


To generate the CS from your domain, use the function:

```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New()
...
// Add level-sets to domain
...
domain->generateCellSet(position, isCellSetAboveSurface)
```
The `position` parameter describes the location of the cell set surface. With the parameter `isCellSetAboveSurface` one can specify whether the Cell-Set should be placed above or below the surface. If the Cell-Set is above the surface it covers all material in the domain and the `position` parameter should be set higher than the highest surface point in the domain. 

## Related Examples

* [Volume Model](https://github.com/ViennaTools/ViennaPS/tree/master/examples/volumeModel)
* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)