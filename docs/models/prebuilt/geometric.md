---
layout: default
title: Geometric Models
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 12
---

# Geometric Models
{: .fs-9 .fw-500}

```c++
#include <psGeometricDistributionModels.hpp>
```
---

Geometric models apply purely geometric transformations to the surface, independent of any particle fluxes or surface chemistry. They are typically used to quickly emulate processes such as isotropic or anisotropic etching and deposition.

Each geometric model is defined by a *surface distribution function*, which specifies how the processed surface is constructed. For instance, an isotropic deposition model uses a *spherical distribution* of constant radius: a sphere is centered at every surface point, and the outermost envelope of all spheres forms the new surface after deposition.

The sign of the distribution size determines whether the model deposits or etches. `SphereDistribution` deposits for a positive radius and etches otherwise. `BoxDistribution` and `CustomSphereDistribution` deposit unless at least one provided size is negative. Anisotropic behavior can be introduced by using non-spherical or directional distributions, such as box-shaped or custom-defined ones. ViennaPS provides three ready-to-use geometric distributions:

* `SphereDistribution` — for isotropic processes
* `BoxDistribution` — for directional or anisotropic processes
* `CustomSphereDistribution` — for user-defined spherical variants

These models can be applied directly as `ProcessModelCPU` objects within a `Process`.

---

#### **Sphere Distribution**

Creates an isotropic geometric model using spherical envelopes of constant radius.
Each surface point is expanded into a sphere of radius `r`, and the new surface is formed by the outermost boundary of all spheres. A positive radius deposits material, while a zero or negative radius creates an etching model.

**Constructor:**

```cpp
SphereDistribution(NumericType radius, LSPtr mask = nullptr)
```

**Parameters:**

| Name     | Type                                                          | Description                                                                       |
| -------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `radius` | `NumericType`                                                 | Radius of the spherical distribution (controls isotropic expansion or shrinkage). |
| `mask`   | `SmartPointer<viennals::Domain<NumericType, D>>` *(optional)* | Optional mask layer that restricts the process to specific regions.               |

**Additional methods:**

```cpp
void addMaskMaterial(const Material material)
void applyToSingleMaterial(const Material material)
```

| Method | Description |
|--------|-------------|
| `addMaskMaterial` | Adds a material that is treated as a mask and is not processed. |
| `applyToSingleMaterial` | Restricts the process to a single material. |

---

#### **Box Distribution**

Creates an anisotropic geometric model based on box-shaped envelopes.
Each surface point is expanded by a rectangular box defined by its half-axis lengths, allowing directional or anisotropic effects. If any half-axis is negative, the model is treated as an etching model; otherwise, it is treated as a deposition model.

**Constructor:**

```cpp
BoxDistribution(const std::array<NumericType, 3> &halfAxes,
                LSPtr mask = nullptr)
```

**Parameters:**

| Name       | Type                                                          | Description                                                          |
| ---------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| `halfAxes` | `std::array<NumericType, 3>`                                  | Half-axis lengths of the box distribution in x, y, and z directions. |
| `mask`     | `SmartPointer<viennals::Domain<NumericType, D>>` *(optional)* | Optional mask layer to confine processing.                           |

**Additional methods:**

```cpp
void addMaskMaterial(const Material material)
void applyToSingleMaterial(const Material material)
```

| Method | Description |
|--------|-------------|
| `addMaskMaterial` | Adds a material that is treated as a mask and is not processed. |
| `applyToSingleMaterial` | Restricts the process to a single material. |

---

#### **Custom Sphere Distribution**

Defines a custom isotropic model where the radius of the spherical distribution varies for each surface point.
Each surface point uses the corresponding radius from the provided list. If any radius is negative, the model is treated as an etching model; otherwise, it is treated as a deposition model. Surface points can be extracted by using the `ToDiskMesh` utility.

**Constructor:**

```cpp
CustomSphereDistribution(const std::vector<NumericType> &radii,
                         LSPtr mask = nullptr)
```

**Parameters:**

| Name    | Type                                                          | Description                                                                                           |
| ------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `radii` | `std::vector<NumericType>`                                    | List of radii for each surface point. |
| `mask`  | `SmartPointer<viennals::Domain<NumericType, D>>` *(optional)* | Optional mask restricting which surface regions are processed.                                        |

**Additional methods:**

```cpp
void addMaskMaterial(const Material material)
```

| Method | Description |
|--------|-------------|
| `addMaskMaterial` | Adds a material that is treated as a mask and is not processed. |

---

Each of these models can be used as input to a `Process` object:

```cpp
auto model = SmartPointer<SphereDistribution<double, 3>>::New(1.0);
Process(domain, model, time).apply();
```

## Related Examples

* [Trench Deposition Geometric](https://github.com/ViennaTools/ViennaPS/tree/master/examples/trenchDepositionGeometric)
