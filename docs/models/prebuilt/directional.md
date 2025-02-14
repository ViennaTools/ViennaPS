---
layout: default
title: Directional Etching
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 3
---

# Directional Etching
{: .fs-9 .fw-500}

```c++
#include <psDirectionalEtching.hpp>
```
---

The Directional Etching Model simulates anisotropic material removal in semiconductor fabrication processes. It models etching where material removal occurs preferentially in a specified direction, influenced by both directional and isotropic velocity components.

```c++
// namespace viennaps
DirectionalEtching(const std::array<NumericType, 3> &direction,
                   NumericType directionalVelocity = 1.,
                   NumericType isotropicVelocity = 0.,
                   const Material mask = Material::Mask, 
                   bool calculateVisibility = true)

DirectionalEtching(const Vec3D<NumericType> &direction,
                   NumericType directionalVelocity,
                   NumericType isotropicVelocity = 0.,
                   const std::vector<Material> &maskMaterials =
                        std::vector<Material>{Material::Mask},
                    bool calculateVisibility = true)

DirectionalEtching(const RateSet &rateSet)
```

| Parameter                 | Description                                                           | Type                  |
|---------------------------|-----------------------------------------------------------------------|-----------------------|
| `direction`               | Direction vector for directional etching.                             | `std::array<NumericType, 3>` |
| `directionalVelocity`     | (Optional) Velocity for directional etching.                          | `NumericType`         |
| `isotropicVelocity`       | (Optional) Isotropic velocity for etching. Default is set to 0.       | `NumericType`         |
| `mask`                    | (Optional) Material used as a mask. Default is set to `Material::Mask`. | `Material`          |
| `calculateVisibility`     | (Optional) Determines whether etching is limited by visibility constraints. Default is set to `true`.        | `bool`          |

A specialized direction can also specified using a `RateSet` with the following parameters:

| Parameter              | Type                        | Description |
|------------------------|---------------------------|-------------|
| `direction`           | `Vec3D<NumericType>`       | The preferred etching direction. |
| `directionalVelocity` | `NumericType`             | The etch rate along the specified direction. |
| `isotropicVelocity`   | `NumericType`             | Velocity applied isotropically. |
| `maskMaterials`       | `std::vector<Material>`   | List of materials used as a mask. |
| `calculateVisibility` | `bool`                    | Determines whether etching is limited by visibility constraints. |

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
// namespace viennaps
RateSet<NumericType> rateSet;
rateSet.direction = Vec3D<NumericType>{0., 0., 1.};
rateSet.directionalVelocity = -1.;
rateSet.isotropicVelocity = 0.;
rateSet.maskMaterials = {Material::Si, Material::SiO2};
rateSet.calculateVisibility = true;

auto model = SmartPointer<DirectionalEtching<NumericType, D>>(rateSet);
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
rateSet = vps.RateSet()
rateSet.direction = [0., 0., 1.]
rateSet.directionalVelocity = -1.
rateSet.isotropicVelocity = 0.
rateSet.maskMaterials = [vps.Material.Si, vps.Material.SiO2]
rateSet.calculateVisibility = True

model = vps.DirectionalEtching(rateSet)
```
</details>
