---
layout: default
title: Material Mapping
parent: Simulation Domain
nav_order: 2
---

# Material Mapping
{: .fs-9 .fw-500 }

---

If specified, each Level-Set in the `psDomain` class is assigned a specific material, which can be used in a process to implement material-specific rates or similar.
The following materials are currently available in the `psMaterial` enum:

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
| `TiN`       | 12    | Titanium Nitride |
| `Cu`        | 13    | Copper       |
| `Polymer`   | 14    | Polymer      |
| `Dielectric`| 15    | Dielectric Material |
| `Metal`     | 16    | Metal        |
| `Air`       | 17    | Air          |
| `GAS`       | 18    | Gas          |

__Example__:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
auto material = psMaterial::Si;
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