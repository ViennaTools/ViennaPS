---
layout: default
title: Material Mapping
parent: Simulation Domain
nav_order: 2
---

# Material Mapping
{: .fs-9 .fw-500 }

---

If specified, each Level-Set in the `Domain` class is assigned a specific material, which can be used in a process to implement material-specific rates or similar.
The following materials are currently available in the `Material` enum:

| Material  | Numeric Value | Description  |
|-----------|-------|--------------|
| `None`      | -1    | Undefined Material  |
| `Mask`      | 0     | Mask Material|
| `Si`        | 1     | Silicon      |
| `SiO2`      | 2     | Silicon Dioxide |
| `Si3N4`     | 3     | Silicon Nitride |
| `SiN`       | 4     | Silicon Nitride |
| `SiON`      | 5     | Silicon Oxynitride |
| `SiC`       | 6     | Silicon Carbide |
| `SiGe`      | 7     | Silicon Germanium |
| `PolySi`    | 8     | Polysilicon  |
| `GaN`       | 9     | Gallium Nitride |
| `W`         | 10    | Tungsten     |
| `Al2O3`     | 11    | Aluminum Oxide |
| `HfO2`       | 12    | Hafnium Oxide |
| `TiN`       | 13    | Titanium Nitride |
| `Cu`        | 14    | Copper       |
| `Polymer`   | 15    | Polymer      |
| `Dielectric`| 16    | Dielectric Material |
| `Metal`     | 17    | Metal        |
| `Air`       | 18    | Air          |
| `GAS`       | 19    | Gas          |

__Example__:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
auto material = Material::Si;
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```c++
material = vps.Material.Si;
```
</details>