---
layout: default
title: Custom Geometry with ViennaLS
parent: Tutorials
nav_order: 5
---

# Custom Geometry with ViennaLS

While `MakeTrench`, `MakeHole`, and `MakePlane` cover many use cases, complex geometries require direct use of ViennaLS — the level set library underlying ViennaPS.

ViennaLS is accessible via `ps.ls` in Python.

## Basic Concepts

A **level set** is an implicit surface representation where the surface is defined as the zero-contour of a signed distance function. ViennaPS stores geometry as a stack of level sets, each representing a material interface.

## Creating Custom Shapes

### Step 1: Get the Grid from the Domain

Custom level sets must share the same grid as your domain:

```python
import viennaps as ps
ps.setDimension(2)

# Create domain
domain = ps.Domain(gridDelta=1.0, xExtent=100.0)
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()

# Get the grid for creating custom level sets
grid = domain.getGrid()
```

### Step 2: Create a Level Set with Geometry

Use `ps.ls.Domain` to create a new level set and `ps.ls.MakeGeometry` to fill it with a shape:

```python
# Create a new level set on the same grid
custom_ls = ps.ls.Domain(grid)

# Create a box (rectangle in 2D)
# Box takes min and max corners: [xMin, yMin], [xMax, yMax]
ps.ls.MakeGeometry(
    custom_ls,
    ps.ls.Box([30.0, 0.0], [70.0, 20.0])
).apply()
```

### Step 3: Insert into Domain

Add the custom level set to the domain with a material:

```python
domain.insertNextLevelSetAsMaterial(custom_ls, ps.Material.Mask)
domain.saveSurfaceMesh("custom_box")
```

## Boolean Operations

Combine multiple shapes using Boolean operations:

```python
# Create first box
box1 = ps.ls.Domain(grid)
ps.ls.MakeGeometry(box1, ps.ls.Box([20.0, 0.0], [40.0, 15.0])).apply()

# Create second box
box2 = ps.ls.Domain(grid)
ps.ls.MakeGeometry(box2, ps.ls.Box([60.0, 0.0], [80.0, 15.0])).apply()

# Union: combine both boxes into one level set
ps.ls.BooleanOperation(box1, box2, ps.ls.BooleanOperationEnum.UNION).apply()

# Now box1 contains both shapes
domain.insertNextLevelSetAsMaterial(box1, ps.Material.SiO2)
```

Available Boolean operations:
*   `UNION`: Combine shapes (logical OR)
*   `INTERSECTION`: Keep only overlapping regions (logical AND)
*   `RELATIVE_COMPLEMENT`: Subtract second from first
*   `INVERT`: Flip inside/outside

## Creating Geometry from Meshes

For complex shapes, you can create geometry from a surface mesh:

```python
# Create a mesh
mesh = ps.ls.Mesh()

# Add nodes (vertices)
mesh.insertNextNode([0.0, 0.0, 0.0])
mesh.insertNextNode([10.0, 0.0, 0.0])
mesh.insertNextNode([10.0, 5.0, 0.0])
mesh.insertNextNode([0.0, 5.0, 0.0])

# Add lines connecting nodes (for 2D)
mesh.insertNextLine([0, 1])
mesh.insertNextLine([1, 2])
mesh.insertNextLine([2, 3])
mesh.insertNextLine([3, 0])

# Convert mesh to level set
custom_ls = ps.ls.Domain(grid)
ps.ls.FromSurfaceMesh(custom_ls, mesh).apply()
```

## 3D Example: Creating Mandrels

A common pattern in advanced patterning (SAQP, etc.):

```python
ps.setDimension(3)

domain = ps.Domain(gridDelta=1.0, xExtent=100.0, yExtent=100.0)
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()
grid = domain.getGrid()

# Create first mandrel
mandrel1 = ps.ls.Domain(grid)
ps.ls.MakeGeometry(
    mandrel1,
    ps.ls.Box([20.0, 0.0, 0.0], [30.0, 100.0, 15.0])  # [xMin,yMin,zMin], [xMax,yMax,zMax]
).apply()

# Create second mandrel
mandrel2 = ps.ls.Domain(grid)
ps.ls.MakeGeometry(
    mandrel2,
    ps.ls.Box([70.0, 0.0, 0.0], [80.0, 100.0, 15.0])
).apply()

# Combine mandrels
ps.ls.BooleanOperation(mandrel1, mandrel2, ps.ls.BooleanOperationEnum.UNION).apply()

# Add to domain
domain.insertNextLevelSetAsMaterial(mandrel1, ps.Material.SiN)
domain.saveSurfaceMesh("mandrels")
```

## Complete Example: Periodic Gratings

Creating a periodic grating pattern:

```python
import viennaps as ps
ps.setDimension(2)

# Domain with periodic boundaries
domain = ps.Domain(
    gridDelta=0.5,
    xExtent=100.0,
    boundary=ps.BoundaryType.PERIODIC_BOUNDARY
)
ps.MakePlane(domain, height=0.0, material=ps.Material.Si).apply()
grid = domain.getGrid()

# Create grating lines
grating = ps.ls.Domain(grid)
line_width = 10.0
pitch = 25.0
num_lines = 4

for i in range(num_lines):
    x_center = pitch * i + pitch / 2
    line = ps.ls.Domain(grid)
    ps.ls.MakeGeometry(
        line,
        ps.ls.Box([x_center - line_width/2, 0.0], [x_center + line_width/2, 20.0])
    ).apply()
    ps.ls.BooleanOperation(grating, line, ps.ls.BooleanOperationEnum.UNION).apply()

domain.insertNextLevelSetAsMaterial(grating, ps.Material.Mask)
domain.saveSurfaceMesh("grating")
```

## See Also

*   [ViennaLS Documentation](https://viennatools.github.io/ViennaLS/) for more level set operations
*   [Domain Functions]({% link domain/functions.md %}) for `insertNextLevelSetAsMaterial` and other domain methods
