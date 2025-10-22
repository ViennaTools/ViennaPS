---
layout: default
title: Material Mapping
parent: Simulation Domain
nav_order: 2
---

# Material Mapping
{: .fs-9 .fw-500 }

---

Each Level-Set in a `Domain` can be assigned a **material** through the `Material` enum.  
Materials are grouped into categories, each with typical density and conductivity properties.  
These properties can be used by process models to determine **material-specific behavior**, such as etch or deposition rates.

---

## Material Categories

| Category | Description |
|-----------|--------------|
| `Generic` | General-purpose or undefined materials |
| `Silicon` | Silicon and silicon-based derivatives |
| `OxideNitride` | Oxides, nitrides, and related compounds |
| `Hardmask` | Carbon-based or organic hardmask layers |
| `Metal` | Metals and conductive layers |
| `Silicide` | Metal-silicon compounds used for contacts |
| `Compound` | Compound semiconductors (III–V, etc.) |
| `TwoD` | 2D or layered materials (e.g. graphene, MoS₂) |
| `TCO` | Transparent conductive oxides |
| `Misc` | Other materials not fitting the above groups |

---

## Material List

| ID | Symbol | Category | Density [g/cm³] | Conductive |
|----|---------|-----------|----------------|-------------|
| 0  | `Mask` | Hardmask | 500.0 | No |
| 1  | `Polymer` | Generic | 1.2 | No |
| 2  | `Air` | Generic | 0.0012 | No |
| 3  | `GAS` | Generic | 0.001 | No |
| 4  | `Dielectric` | Generic | 2.2 | No |
| 5  | `Metal` | Metal | 7.5 | Yes |
| 6  | `Undefined` | Generic | 0.0 | No |
| 10–21 | Silicon and derivatives (`Si`, `PolySi`, `aSi`, `SiGe`, `SiC`, `SiN`, `Si3N4`, `SiON`, `SiCN`, `SiBCN`, `SiCOH`, `SiOCN`) | Silicon / OxideNitride | 1.9–4.0 | No |
| 30–40 | Oxides and nitrides (`SiO2`, `Al2O3`, `HfO2`, `ZrO2`, `TiO2`, `Y2O3`, `La2O3`, `AlN`, `Ta2O5`, `BN`, `hBN`) | OxideNitride | 2.1–9.7 | No |
| 50–60 | Hardmask / organics (`C`, `aC`, `SOC`, `SOG`, `BPSG`, `PSG`, `SiLK`, `ARC`, `PMMA`, `PHS`, `HSQ`) | Hardmask / OxideNitride | 1.0–2.2 | No |
| 70–90 | Metals (`W`, `Cu`, `Co`, `Ru`, `Ni`, `Pt`, `Ta`, `TaN`, `Ti`, `TiN`, `Mo`, `Ir`, `Rh`, `Pd`, `RuTa`, `CoW`, `NiW`, `TiAlN`, `Mn`, `MnO`, `MnN`) | Metal | 4.5–22.6 | Mostly Yes |
| 100–102 | Silicides (`WSi2`, `TiSi2`, `MoSi2`) | Silicide | 4.0–9.3 | Yes |
| 110–116 | Compound semiconductors (`Ge`, `GaN`, `GaAs`, `InP`, `InGaAs`, `SiGaN`, `SiOCH`) | Compound / OxideNitride | 1.8–6.15 | No |
| 130–135 | 2D materials (`Graphene`, `MoS2`, `WS2`, `WSe2`, `VO2`, `GST`) | TwoD | 2.2–9.3 | Mostly No |
| 150–152 | Transparent conductors (`ITO`, `ZnO`, `AZO`) | TCO | 5.5–7.1 | Yes |
| 170–175 | Hardmask aliases (`SiON_HM`, `SiN_HM`, `SiC_HM`, `TiO`, `ZrO`, `SiO2_HM`) | Hardmask / Misc | 2.2–5.2 | No |

---

## Accessing Material Information

Each material has associated metadata accessible through helper functions:

| Function | Description | Example |
|-----------|-------------|----------|
| `to_string_view(Material)` | Returns the material name | `"SiO2"` |
| `categoryOf(Material)` | Returns the `MaterialCategory` | `MaterialCategory::OxideNitride` |
| `density(Material)` | Returns the density in g/cm³ | `2.2` |
| `isConductive(Material)` | Returns true if conductive | `false` |

---

## Examples

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```cpp
auto material = ps::Material::Si;
auto category = ps::categoryOf(material);
auto rho = ps::density(material);
auto isCond = ps::isConductive(material);
````

</details>

---

## Notes

* The material list now includes **over 170 entries** covering silicon, oxides, nitrides, metals, silicides, 2D materials, and transparent conductors.
* Each material entry includes physical and electrical properties used in **process modeling and analysis**.
* The enumeration and metadata are defined in `psMaterials.hpp`.


