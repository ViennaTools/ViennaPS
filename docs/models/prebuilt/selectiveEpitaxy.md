---
layout: default
title: Selective Epitaxy
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 11
---

# Selective Epitaxy
{: .fs-9 .fw-500}

```c++
#include <psSelectiveEpitaxy.hpp>
```
---

The **SelectiveEpitaxy** model simulates epitaxial growth that is restricted to specific materials and regions, while other areas remain masked.
It supports crystal-orientation–dependent growth rates, allowing for more accurate modeling of faceted epitaxial fronts.
The process is “selective” because growth occurs only on materials explicitly marked as epitaxy materials; all other surfaces are treated as masked.

This model is useful for simulating **selective area growth (SAG)** in semiconductor fabrication, where epitaxy is limited to patterned openings while masked regions remain untouched.

During simulation:

* The model determines where epitaxy is allowed based on the user-specified material list.
* Growth rates depend on the surface normal’s orientation, interpolating between `{111}` and `{100}` growth rates.
* Non-epitaxy materials act as masks, preventing growth in those areas.
* Internally, the model creates a mask layer using Boolean operations on the domain geometry before starting the epitaxial growth.

---

## Constructor Parameters

| Parameter    | Type                                       | Description                                                                          | Units / Range             | Default |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------- | ------- |
| `materialRates` | `std::vector<std::pair<Material, double>>` | List of materials where epitaxy is allowed, with corresponding rate scaling factors. | Scaling factor (unitless) | —       |
| `rate111`       | `double`                                   | Growth rate for `{111}` crystal orientations.                                        | User units (e.g., µm/s)   | 0.5     |
| `rate100`       | `double`                                   | Growth rate for `{100}` crystal orientations.                                        | User units (e.g., µm/s)   | 1.0     |

---

## Example Usage

**Basic usage** – Epitaxy on Si only, with default `{111}` and `{100}` rates:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
{% raw %}
```cpp
using namespace viennaps;

std::vector<std::pair<Material, double>> epiMaterials = {
    {Material::Si, 1.0}
};

auto epi = SmartPointer<SelectiveEpitaxy<double, 3>>::New(epiMaterials);
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
import viennaps as vps

epiMaterials = [
    (vps.Material.Si, 1.0)
]

model = vps.SelectiveEpitaxy(
    materialRates=epiMaterials
)
```
{% endraw %}
</details>

**Custom orientation rates** – Faster `{100}` growth, slower `{111}` growth:

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
{% raw %}
```cpp
using namespace viennaps;

std::vector<std::pair<Material, double>> epiMaterials = {
    {Material::Si, 1.0}
};

double r111 = 0.3; // µm/s
double r100 = 1.2; // µm/s

auto epi = SmartPointer<SelectiveEpitaxy<double, 3>>::New(epiMaterials, r111, r100);
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
import viennaps as vps

epiMaterials = [
    (vps.Material.Si, 1.0)
]
model = vps.SelectiveEpitaxy(
    materialRates=epiMaterials,
    rate111=0.3,  # µm/s
    rate100=1.2   # µm/s
)
```
{% endraw %}
</details>

**Tips:**

* The topmost material in the domain **must** be an epitaxy material; otherwise, an error is logged.
* At least two level sets are required in the domain for selective epitaxy.
* Mask layers are automatically generated during initialization to block growth in non-epitaxy regions.

## Related Examples

* [Selective Epitaxy](https://github.com/ViennaTools/ViennaPS/tree/master/examples/selectiveEpitaxy)
