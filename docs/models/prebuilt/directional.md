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
using RateMap = MaterialValueMap<std::pair<NumericType, NumericType>>;

DirectionalProcess(const Vec3D<NumericType> &direction,
                   RateMap materialRates)

DirectionalProcess(const Vec3D<NumericType> &direction,
                   std::unordered_map<Material,
                                      std::pair<NumericType, NumericType>>
                       materialRates,
                   NumericType defaultDirectionalRate = 0.,
                   NumericType defaultIsotropicRate = 0.)

DirectionalProcess(const Vec3D<NumericType> &direction,
                   NumericType directionalVelocity,
                   NumericType isotropicVelocity = 0.,
                   Material maskMaterial = Material::Mask,
                   bool calculateVisibility = true)

DirectionalProcess(const Vec3D<NumericType> &direction,
                   NumericType directionalVelocity,
                   NumericType isotropicVelocity,
                   const std::vector<Material> &maskMaterials,
                   bool calculateVisibility = true)

DirectionalProcess(const RateSet &rateSet)

DirectionalProcess(std::vector<RateSet> rateSets)
```

| Parameter                 | Description                                       | Type                  |
|---------------------------|------------------------------------------------|-----------------------|
| `direction`               | Direction vector for directional process.         | `Vec3D<NumericType>` |
| `materialRates`           | Material-specific directional and isotropic rates. The pair stores `{directionalRate, isotropicRate}`. | `RateMap` or `std::unordered_map<Material, std::pair<NumericType, NumericType>>` |
| `defaultDirectionalRate`  | Directional fallback rate for materials not listed in `materialRates`. | `NumericType` |
| `defaultIsotropicRate`    | Isotropic fallback rate for materials not listed in `materialRates`. | `NumericType` |
| `directionalVelocity`     | (Optional) Velocity for directional process.      | `NumericType`         |
| `isotropicVelocity`       | (Optional) Isotropic velocity. Default is set to 0.       | `NumericType`         |
| `maskMaterial`            | (Optional) Material used as a mask. Default is set to `Material::Mask`. | `Material`          |
| `maskMaterials`           | Materials used as masks for the rate set. | `std::vector<Material>` |
| `calculateVisibility`     | (Optional) Determines whether the process is limited by visibility constraints. Default is set to `true`.        | `bool`          |

A specialized direction can also be specified using one or more `RateSet` objects with the following parameters:

| Parameter              | Type                        | Description |
|------------------------|---------------------------|-------------|
| `direction`           | `Vec3D<NumericType>`       | The preferred process direction. |
| `directionalVelocity` | `NumericType`             | The etch rate along the specified direction. |
| `isotropicVelocity`   | `NumericType`             | Velocity applied isotropically. |
| `maskMaterials`       | `std::vector<Material>`   | List of materials used as a mask. |
| `calculateVisibility` | `bool`                    | Determines whether the process is limited by visibility constraints. It is automatically disabled when the direction is zero or the directional velocity is zero. |

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

auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(rateSet);
```
</details>

<details markdown="1">
<summary markdown="1">
C++ Material Rates
{: .label .label-blue }
</summary>
```c++
std::unordered_map<Material, std::pair<NumericType, NumericType>> rates = {
    {Material::Si, {-1.0, 0.0}},
    {Material::SiO2, {-0.2, 0.0}},
    {Material::Mask, {0.0, 0.0}},
};

auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
    Vec3D<NumericType>{0., 0., 1.}, rates);
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

<details markdown="1">
<summary markdown="1">
Python Material Rates
{: .label .label-green }
</summary>
```python
rates = {
    vps.Material.Si: (-1.0, 0.0),
    vps.Material.SiO2: (-0.2, 0.0),
    vps.Material.Mask: (0.0, 0.0),
}

model = vps.DirectionalProcess([0., 0., 1.], rates)
```
</details>
