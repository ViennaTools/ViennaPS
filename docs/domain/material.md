---
layout: default
title: Material Mapping
parent: Simulation Domain
nav_order: 2
---

# Material Mapping
{: .fs-9 .fw-500 }

---

Each Level-Set in a `Domain` can be assigned a **material** through the `Material` type.  
Materials are grouped into categories, each with typical density and conductivity properties.  
These properties can be used by process models to determine **material-specific behavior**, such as etch or deposition rates.

---

## 1. Design Goals

The system separates **fixed built-in materials** from **runtime custom materials**, while exposing a single type (`viennaps::Material`) throughout APIs.

Key properties:

- Built-ins are compile-time known and stable by ID/name.
- Custom materials are registered at runtime by name.
- Most APIs only deal with `Material`, not raw integers.
- Legacy numeric material IDs are still interoperable where needed.

## 2. Core Types and Responsibilities

### 2.1 `BuiltInMaterial`

- Enum of fixed materials (for example `Mask`, `Si`, `Polymer`, ...).
- Backed by a generated table (`kBuiltInMaterialTable`) that provides:
  - canonical material name (`std::string_view`)
  - category (`MaterialCategory`)
  - density (`density_gcm3`)
  - conductivity flag
  - display color (`colorHex`)
- Conversion helpers:
  - `builtInMaterialToString(...)`
  - `tryBuiltInMaterialFromString(...)`
  - `builtInMaterialFromString(...)`

### 2.2 `Material`

`Material` is the unified handle used in process/domain APIs.

Internal representation:

- `Kind::BuiltIn` + built-in enum value
- `Kind::Custom` + custom numeric ID

Static constants are provided for all built-ins, for example:

- `Material::Mask`
- `Material::Si`
- `Material::Polymer`

Legacy ID mapping convention:

- built-in IDs occupy `[0, kBuiltInMaterialMaxId]`
- custom material legacy IDs start at `kBuiltInMaterialMaxId + 1`

### 2.3 `MaterialRegistry`

Singleton registry for runtime custom materials.

Main behavior:

- `registerMaterial(name)`:
  - returns built-in `Material` if `name` matches a built-in
  - otherwise creates/reuses a custom material entry
- `hasMaterial(name/material)`, `findMaterial(name)`, `getMaterial(name)`
- `getName(material)` to recover canonical name
- `getInfo(material)` / `setInfo(material, info)`
- `customMaterialCount()`

Notes:

- Built-ins are always considered present.
- Custom material metadata defaults to category `Generic`, density `0.0`, non-conductive, color `0xffffff`.

### 2.4 `MaterialMap`

`MaterialMap` wraps ViennaLS layer-material mapping and provides convenience helpers:

- `mapToMaterial(int)` and templated `mapToMaterial(T)`
- `isMaterial(...)`
- `isHardmask(...)`
- `toString(Material|int)`
- `fromString(name)`:
  - resolves built-in by name, or
  - registers/returns custom via `MaterialRegistry`

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```cpp
#include <materials/psMaterial.hpp>
#include <materials/psMaterialRegistry.hpp>
#include <materials/psMaterialValueMap.hpp>

using namespace viennaps;

Material si = Material::Si; // built-in
Material custom = MaterialMap::fromString("CustomFoo");

MaterialValueMap<double> rates;
rates.setDefault(0.0);
rates.set(Material::Mask, 0.0);
rates.set(si, 0.5);
rates.set(custom, 1.0);

double rSi = rates.get(Material::Si);
double rCustom = rates.get(custom);
```

Name-based domain insertion:

```cpp
domain->insertNextLevelSetAsMaterial(levelSet, "CustomMaterial");
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
si = ps.Material(ps.BuiltInMaterial.Si)
custom = ps.MaterialMap.fromString("CustomFoo")

rates = ps.MaterialValueMap()
rates.setDefault(0.0)
rates.set(ps.Material.Mask, 0.0)
rates.set(si, 0.5)
rates.set(custom, 1.0)

print(rates.get(custom))
```
</details>

