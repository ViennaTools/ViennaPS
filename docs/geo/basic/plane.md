---
layout: default
title: Plane Geometry
parent: Basic Geometries
grand_parent: Creating a Geometry
nav_order: 1
---

# Plane Geometry
{: .fs-9 .fw-500 }

```c++
#include <psMakePlane.hpp>
```

---

The `MakePlane` class offers a straightforward approach to generate a plane as a level-set within your domain. This utility is useful for crafting substrates with any material. You have the flexibility to append the plane to an existing geometry or create a new one. In 3D, the plane is generated with a normal direction in the positive z direction, while in 2D, it is oriented in the positive y direction. The plane is centered around the origin, with the total specified extent and height. Additionally, you can opt for a periodic boundary in the x and y directions.

```c++
// namespace viennaps
// New geometry
MakePlane(DomainType domain, 
         const NumericType gridDelta,
         const NumericType xExtent, 
         const NumericType yExtent,
         const NumericType height, 
         const bool periodicBoundary = false,
         const Material material = Material::None)

// Add to existing geometry
MakePlane(DomainType domain, NumericType height = 0.,
          const Material material = Material::None)
```

Depending on the specific constructor invoked for the plane-builder, the behavior varies: the domain may be cleared, and a new plane inserted, or the plane can be added to the existing geometry in the domain. A detailed description of the parameters follows:

| Parameter    | Description                                      | Type |
|:-------------|:-------------------------------------------------|:-------------------------------------------|
| `domain`       | The `Domain` object passed in a smart pointer. | `SmartPointer<Domain<NumericType, D>>` |
| `gridDelta`    | Represents the grid spacing or resolution used in the simulation.   | `NumericType`  |
| `xExtent`           | Defines the extent of the plane geometry in the _x_-direction.     | `NumericType`   |
| `yExtent`           | Defines the extent of the plane geometry in the _y_-direction. | `NumericType`  |
| `height`      | Sets the position of the plane in y(2D)/z(3D) direction. | `NumericType` |
| `periodicBoundary` | (Optional) If set to true, enables periodic boundaries in both x and y directions. Default is set to false. | `bool` |
| `material` | (Optional) Specifies the material used for the plane. Default is set to `Material::None`. | `Material` |


__Example usage__:

* Creating a new domain: 

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps
auto domain = SmartPointer<Domain<NumericType, D>>::New();
MakePlane<NumericType, D>(domain, 0.5, 10.0, 10.0, 0.0, false,
                          Material::Si).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
domain = vps.Domain()
vps.MakePlane(domain=domain,
              gridDelta=0.5,
              xExtent=10.0,
              yExtent=10.0,
              height=0.0,
              periodicBoundary=False,
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
auto domain = SmartPointer<Domain<NumericType, D>>::New();
psMakePlane<NumericType, D>(domain, 10.0, Material::Si).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
domain = vps.Domain()
vps.MakePlane(domain=domain,
              height=0.0,
              material=vps.Material.Si,
             ).apply()
```
</details>