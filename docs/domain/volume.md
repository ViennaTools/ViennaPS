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

The cell set supports spatial queries and cell-neighbor relations. Call
`buildNeighborhood()` after generation when preparing a volume process that
uses neighboring cells.


## How to use the Cell-Set
{: .lh-default}


To generate the CS from your domain, use the function:

```c++
void generateCellSet(NumericType position, Material coverMaterial,
                     bool isAboveSurface = false,
                     bool withEmbeddedBoundaries = false);
```
`position` places the bounding plane along the primary direction (y in 2D,
z in 3D). With `isAboveSurface = true`, place it above the highest surface
point to include the material stack and a cover region, such as air.
`coverMaterial` assigns the material of that cover region.

For implantation, include a lower substrate plane deep enough to contain the
profile and an upper surface plane, as shown in the
[implantation example]({% link models/prebuilt/ionImplantation.md %}). Then:

```c++
domain->generateCellSet(20.0, Material::Air, true, true);
domain->getCellSet()->buildNeighborhood();
```

```python
domain.generateCellSet(20.0, vps.Material.Air, True, True)
domain.getCellSet().buildNeighborhood()
```

`withEmbeddedBoundaries` adds sub-grid boundary information used to improve
tilted implantation at material interfaces. Generate the cell set before
implanting, then reuse it for annealing so the concentration and damage fields
remain available. Regeneration reconstructs the volume representation.

Write volume process fields with `domain->getCellSet()->writeVTU("volume")`
(Python: `domain.getCellSet().writeVTU("volume")`).
`Domain::saveVolumeMesh` instead exports a visualization of the level-set
geometry; see [Volume Mesh]({% link output/volume.md %}).

## Related Examples

* [ViennaCS](https://github.com/ViennaTools/ViennaCS/tree/main/examples)
* [Oxide Regrowth](https://github.com/ViennaTools/ViennaPS/tree/master/examples/oxideRegrowth)
* [Ion Implantation and Anneal](https://github.com/ViennaTools/ViennaPS/tree/master/examples/ionImplantation)
* [PN Junction](https://github.com/ViennaTools/ViennaPS/tree/master/examples/pnJunction)
