---
layout: default
title: Geometry Extrusion and Slicing
parent: Miscellaneous
nav_order: 2
---

# Extrude and Slice Geometries between 2D and 3D
{: .fs-9 .fw-500}

---

ViennaPS provides two utility functions for converting between 2D and 3D geometries:

- **Extrude:** Converts a 2D domain into a 3D domain by extending it along a chosen axis.  
- **Slice:** Extracts a 2D cross-section from a 3D domain along a specified axis or plane.

These tools simplify workflows that involve process simulations requiring both 2D and 3D representations.

---

## Extrude

Extrudes a 2D domain into 3D.  
Users can define the extrusion axis, the extent (min/max positions), and boundary conditions for the new dimension.

### Example usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```c++
ps::Extrude<double>(
    domain2D, domain3D,
    {0., 1.}, // min and max extent in the extruded dimension
    2,        // extrude along z-axis (0=x, 1=y, 2=z)
    {viennals::BoundaryConditionEnum::REFLECTIVE,
     viennals::BoundaryConditionEnum::REFLECTIVE,
     viennals::BoundaryConditionEnum::INFINITE_BOUNDARY}
).apply();
````

</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>

```python
import viennaps as vps

vps.Extrude(
    domain2d, domain3d,
    extent=(0.0, 1.0),
    axis=2,  # extrude along z-axis
    boundary_conditions=[
        vps.BoundaryCondition.REFLECTIVE,
        vps.BoundaryCondition.REFLECTIVE,
        vps.BoundaryCondition.INFINITE_BOUNDARY
    ]
).apply()
```

</details>

---

## Slice

Creates a 2D cross-section from a 3D domain along a specified axis and position.
This is useful for inspecting or analyzing a single layer or cut plane from a 3D simulation.

### Example usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```c++
ps::Slice<double>(
    domain3D, domain2D,
    2,    // slice along z-axis (0=x, 1=y, 2=z)
    0.5   // normalized slice position (between min/max domain bounds)
).apply();
```

</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>

```python
import viennaps as vps

vps.Slice(
    domain3d, domain2d,
    axis=2,       # slice along z-axis
    position=0.5  # relative position within domain bounds
).apply()
```

</details>

---

## Notes

* `Extrude` and `Slice` preserve material and level-set information during conversion.
* Boundary conditions in `Extrude` define how the new dimension behaves during subsequent simulations.


