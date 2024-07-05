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

Details Coming soon
{: .label .label-yellow}

```c++
// namespace viennaps
DirectionalEtching(const std::array<NumericType, 3> &direction,
                   const NumericType directionalVelocity = 1.,
                   const NumericType isotropicVelocity = 0.,
                   const Material mask = Material::Mask)
```

| Parameter                 | Description                                                           | Type                  |
|---------------------------|-----------------------------------------------------------------------|-----------------------|
| `direction`               | Direction vector for directional etching.                             | `std::array<NumericType, 3>` |
| `directionalVelocity`     | (Optional) Velocity for directional etching. Default is set to 1.     | `NumericType`         |
| `isotropicVelocity`       | (Optional) Isotropic velocity for etching. Default is set to 0.       | `NumericType`         |
| `mask`                    | (Optional) Material used as a mask. Default is set to `Material::Mask`. | `Material`          |
