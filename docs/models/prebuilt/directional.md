---
layout: default
title: Directional Process
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 2
---

# Directional Process
{: .fs-9 .fw-500}

```c++
#include <psDirectionalProcess.hpp>
```
---

The directional process model emulates anisotropic material removal or deposition. It models etching where material removal occurs preferentially in a specified direction, influenced by both directional and isotropic velocity components.

```c++
// namespace viennaps
DirectionalProcess(const std::array<NumericType, 3> &direction,
                   NumericType directionalVelocity = 1.,
                   NumericType isotropicVelocity = 0.,
                   const Material mask = Material::Mask, 
                   bool calculateVisibility = true)

DirectionalProcess(const Vec3D<NumericType> &direction,
                   NumericType directionalVelocity,
                   NumericType isotropicVelocity = 0.,
                   const std::vector<Material> &maskMaterials =
                        std::vector<Material>{Material::Mask},
                    bool calculateVisibility = true)

DirectionalProcess(const RateSet &rateSet)
```

| Parameter                 | Description                                       | Type                  |
|---------------------------|------------------------------------------------|-----------------------|
| `direction`               | Direction vector for directional process.         | `std::array<NumericType, 3>` |
| `directionalVelocity`     | (Optional) Velocity for directional process.      | `NumericType`         |
| `isotropicVelocity`       | (Optional) Isotropic velocity. Default is set to 0.       | `NumericType`         |
| `mask`                    | (Optional) Material used as a mask. Default is set to `Material::Mask`. | `Material`          |
| `calculateVisibility`     | (Optional) Determines whether the process is limited by visibility constraints. Default is set to `true`.        | `bool`          |

A specialized direction can also specified using a `RateSet` with the following parameters:

| Parameter              | Type                        | Description |
|------------------------|---------------------------|-------------|
| `direction`           | `Vec3D<NumericType>`       | The preferred process direction. |
| `directionalVelocity` | `NumericType`             | The etch rate along the specified direction. |
| `isotropicVelocity`   | `NumericType`             | Velocity applied isotropically. |
| `maskMaterials`       | `std::vector<Material>`   | List of materials used as a mask. |
| `calculateVisibility` | `bool`                    | Determines whether the process is limited by visibility constraints. |

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
rateSet.directionalVelocity = 1.;
rateSet.isotropicVelocity = 0.;
rateSet.maskMaterials = {Material::Si, Material::SiO2};
rateSet.calculateVisibility = true;

auto model = SmartPointer<DirectionalProcess<NumericType, D>>(rateSet);
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
rateSet.directionalVelocity = 1.
rateSet.isotropicVelocity = 0.
rateSet.maskMaterials = [vps.Material.Si, vps.Material.SiO2]
rateSet.calculateVisibility = True

model = vps.DirectionalProcess(rateSet)
```
</details>
