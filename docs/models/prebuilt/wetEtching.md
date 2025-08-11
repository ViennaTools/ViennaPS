---
layout: default
title: Wet Etching
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 10
---

# Wet Etching
{: .fs-9 .fw-500}

```c++
#include <psWetEtching.hpp>
```
---

The **WetEtching** model simulates crystal-orientation–dependent etching processes, commonly used to describe anisotropic wet chemical etching of crystalline materials (e.g., KOH etching of silicon).
The etch rate depends on the orientation of the local surface normal with respect to the crystal directions `{100}`, `{110}`, `{111}`, and `{311}`.
For each allowed etching material, orientation-specific rates can be specified, enabling accurate modeling of material-selective and anisotropic etching behaviors.

Two constructor variants are provided:

* **Simple mode** – specify only the materials to be etched and their global rates; default crystal directions and rates for each orientation are used.
* **Advanced mode** – specify custom crystal direction vectors and per-orientation rates, along with the etching materials.

During simulation:

* The local surface normal is compared to the crystal orientation vectors.
* The velocity field is computed analytically from the given orientation-dependent rates.
* Only materials specified in the `materials` list are etched; others are unaffected.

---

## Constructor Parameters

| Parameter      | Type                                       | Description                                                   | Units / Range                   | Default (simple mode)       |
| -------------- | ------------------------------------------ | ------------------------------------------------------------- | ------------------------------- | --------------------------- |
| `materialRates`   | `std::vector<std::pair<Material, double>>` | List of materials and their global etch rate scaling factors. | Rate in user units (e.g., µm/s) | —                           |
| `direction100` | `Vec3D<double>`                            | Crystal direction vector for the `{100}` family.              | Normalized vector               | `[0.7071, 0.7071, 0]` (3D)  |
| `direction010` | `Vec3D<double>`                            | Crystal direction vector for the `{010}` family.              | Normalized vector               | `[-0.7071, 0.7071, 0]` (3D) |
| `rate100`   | `double`                                   | Etch rate along `{100}` crystal direction.                    | µm/s                            | 0.0166667                   |
| `rate110`   | `double`                                   | Etch rate along `{110}` crystal direction.                    | µm/s                            | 0.0309167                   |
| `rate111`   | `double`                                   | Etch rate along `{111}` crystal direction.                    | µm/s                            | 0.000121667                 |
| `rate311`   | `double`                                   | Etch rate along `{311}` crystal direction.                    | µm/s                            | 0.0300167                   |

---

## Notes

* The **simple mode** automatically sets `direction100` and `direction010` to default cubic crystal orientation vectors for either 2D or 3D simulations.
* All rates (`rate100`, `rate110`, `rate111`, `rate311`) are specified in simulation length units per second and are scaled by the per-material rate factor from `materialRates`.
* Multiple materials can be etched in a single process, each with its own scaling factor.
* Materials not listed in `materialRates` are treated as non-etchable (mask materials).

---

## Example Usage

**Simple mode** – Specify only the materials to be etched and their rate scaling factors:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
{% raw %}
```cpp
using namespace viennaps;

// Etch Si at 1× default rate, SiO2 at 0.5× default rate
std::vector<std::pair<Material, double>> etchMaterials = {
    {Material::Si, 1.0},
    {Material::SiO2, 0.5}
};

auto wetEtch = SmartPointer<WetEtching<double, 3>>::New(etchMaterials);
```
{% endraw %}
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
{% raw %}
```python
...
etchMaterials = [
    (vps.Material.Si, 1.0),
    (vps.Material.SiO2, 0.5)
]

model = vps.WetEtching(etchMaterials)
...
```
{% endraw %}
</details>

**Advanced mode** – Specify custom crystal directions and per-orientation rates:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
{% raw %}
```cpp
using namespace viennaps;

Vec3D<double> dir100 = {1.0, 0.0, 0.0};
Vec3D<double> dir010 = {0.0, 1.0, 0.0};

// Orientation-specific rates in µm/s
double r100 = 0.02;
double r110 = 0.04;
double r111 = 0.0002;
double r311 = 0.03;

std::vector<std::pair<Material, double>> etchMaterials = {
    {Material::Si, 1.0}
};

auto wetEtch = SmartPointer<WetEtching<double, 3>>::New(
    dir100, dir010, r100, r110, r111, r311, etchMaterials
);
```
{% endraw %}
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
{% raw %}
```python
...
etchMaterials = [
    (vps.Material.Si, 1.0),
]

model = vps.WetEtching(
    direction100=[1.0, 0.0, 0.0],
    direction010=[0.0, 1.0, 0.0],
    rate100=0.02,
    rate110=0.04,
    rate111=0.0002,
    rate311=0.03,
    materialRates=etchMaterials
)
...
```
{% endraw %}
</details>

**Tips:**

* Use **simple mode** for standard cubic crystal orientations and default rates.
* Use **advanced mode** when simulating non-standard orientations or when precise orientation rates are available from experiments.
* Units for rates should match the simulation’s spatial and temporal scaling (e.g., µm/s).

---

## Related Examples

* [Cantilever Wet Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/cantileverWetEtching)
