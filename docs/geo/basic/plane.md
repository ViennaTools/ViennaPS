---
layout: default
title: Plane Geometry
parent: Geometry Builders
grand_parent: Creating a Geometry
nav_order: 1
---

# Plane Geometry
{: .fs-9 .fw-500 }

```c++
#include <geometries/psMakePlane.hpp>
```

---

The `MakePlane` class offers a straightforward approach to generate a plane as a level-set within your domain. This utility is useful for crafting substrates with any material. You have the flexibility to append the plane to an existing geometry or create a new one. In 3D, the plane is generated with a normal direction in the positive z direction, while in 2D, it is oriented in the positive y direction. The plane is centered around the origin, with the total specified extent and height. Additionally, you can opt for a periodic boundary in the x and y directions.

```c++
// namespace viennaps
MakePlane(psDomainType domain, 
          NumericType baseHeight = 0.,
          Material material = Material::Si, 
          bool addToExisting = false)

MakePlane(psDomainType domain, 
          NumericType gridDelta, 
          NumericType xExtent,
          NumericType yExtent, 
          NumericType baseHeight,
          bool periodicBoundary = false, 
          Material material = Material::Si)
      
```

Depending on the specific constructor invoked for the plane-builder, the behavior varies: the domain may be cleared, and a new plane inserted, or the plane can be added to the existing geometry in the domain. A detailed description of the parameters follows:

| Parameter           | Type           | Description  | Applicable Constructor |
|---------------------|---------------|--------------|------------------------|
| `domain`           | `psDomainType` | The simulation domain. | Both |
| `baseHeight`       | `NumericType`  | Height at which the plane is placed (default: `0.`). | Both |
| `material`         | `Material`     | Material of the plane (default: `Material::Si`). | Both |
| `addToExisting`    | `bool`         | If `true`, the plane is added to an existing geometry instead of creating a new one (default: `false`). | First constructor only |
| `gridDelta`        | `NumericType`  | Grid spacing in the simulation domain. | Second constructor only |
| `xExtent`         | `NumericType`  | Extent of the domain in the x-direction. | Second constructor only |
| `yExtent`         | `NumericType`  | Extent of the domain in the y-direction. | Second constructor only |
| `periodicBoundary` | `bool`         | If `true`, enables periodic boundary conditions (default: `false`). | Second constructor only |

> **Note**:  
> - The first constructor **requires** that the domain is already configured and allows adding a plane to an existing geometry.  
> - The second constructor creates a new geometry and sets up the domain using `gridDelta`, `xExtent`, and `yExtent`.  

__Example usage__:

* Creating a new domain: 

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps
auto domain = SmartPointer<Domain<NumericType, D>>::New(0.5, 10., 10., BoundaryType::REFLECTIVE_BOUNDARY);
MakePlane<NumericType, D>(domain, 0.0, Material::Si).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
domain = vps.Domain(gridDelta=0.5, xExtent=10.0, yExtent=10.0, boundaryType=vps.BoundaryType.REFLECTIVE_BOUNDARY)
vps.MakePlane(domain=domain,
              baseHeight=0.0,
              material=vps.Material.Si,
             ).apply()
```
</details>

* Adding plane to existing domain

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
MakePlane<NumericType, D>(domain, 10.0, Material::Si, true).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
vps.MakePlane(domain=domain,
              height=0.0,
              material=vps.Material.Si,
              addToExisting=True
             ).apply()
```
</details>