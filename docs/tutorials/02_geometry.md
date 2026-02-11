---
layout: default
title: Geometry Creation
parent: Tutorials
nav_order: 2
---

# Geometry Creation

Before running any process, you must define the simulation domain and the initial geometry.

## The Domain

The `ps.Domain` class holds the level set data representing your geometry. It requires a grid resolution (`gridDelta`) and physical extents.

```python
import viennaps as ps
ps.setDimension(2)

# 2D domain: grid spacing 0.5, width 50
domain = ps.Domain(gridDelta=0.5, xExtent=50.0)
```

For 3D simulations:
```python
ps.setDimension(3)

# 3D domain: grid spacing 0.5, xExtent 50, yExtent 50
domain_3d = ps.Domain(gridDelta=0.5, xExtent=50.0, yExtent=50.0)
```

### Boundary Conditions

You can also specify boundary conditions (periodic, reflective, or infinite):

```python
# Periodic boundaries in x, infinite in vertical direction
domain = ps.Domain(
    gridDelta=0.5,
    xExtent=50.0,
    boundary=ps.BoundaryType.PERIODIC_BOUNDARY
)
```

## Geometric Primitives

ViennaPS provides helper classes to create common initial shapes. These modify the `ps.Domain` in place.

### MakePlane
Creates a flat surface at a specific height, filling everything below with the given material.

```python
# Create a Silicon substrate filling everything below y=0
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()
```

### MakeTrench
Creates a trench structure in a substrate.

```python
ps.MakeTrench(
    domain=domain,
    trenchWidth=20.0,
    trenchDepth=15.0,
    maskHeight=5.0,       # Height of the mask layer above the surface
    trenchTaperAngle=0.0  # Vertical sidewalls (degrees)
).apply()
```

### MakeHole
For 3D simulations, creates a cylindrical or tapered hole.

```python
ps.setDimension(3)
domain_3d = ps.Domain(gridDelta=0.5, xExtent=50.0, yExtent=50.0)

ps.MakeHole(
    domain=domain_3d,
    holeRadius=10.0,
    holeDepth=0.0,        # 0 = mask only (no pre-etched hole)
    maskHeight=5.0
).apply()
```

## Saving Results

You can save the geometry at any point to visualize it.

*   `saveSurfaceMesh("filename")`: Saves the surface mesh as `.vtp` (efficient for visualization).
*   `saveVolumeMesh("filename")`: Saves the full volume mesh as `.vtu` (includes material info in bulk).

```python
domain.saveSurfaceMesh("my_geometry")  # Creates my_geometry.vtp
domain.saveVolumeMesh("my_geometry")   # Creates my_geometry_volume.vtu
```

{: .note }
The file extension is added automatically. Material IDs are stored under the `"MaterialIds"` label in exported meshes.
