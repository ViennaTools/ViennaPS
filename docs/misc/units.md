---
layout: default
title: Units
parent: Miscellaneous
nav_order: 6
---

# Units
{: .fs-9 .fw-500}

---

Physical models, like the SF6O2 or Fluorocarbon etching models, require the user to specify the units of the input parameters. The user must set the length and time units before creating a model using the `units` module. The units are global parameters and are used by models that convert rates through `units`. Some volume models use explicit units, as described below.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
using namespace viennaps;
units::Length::setUnit(units::Length::NANOMETER);
units::Time::setUnit(units::Time::MINUTE);

// the units can also be specified using strings
units::Length::setUnit("nm"); // or "nanometer"
units::Time::setUnit("min"); // or "minute"
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
# in Python strings have to be used to set the units
vps.Length.setUnit("nm") # or "nanometer"
vps.Time.setUnit("min") # or "minute"
```
</details>

## Volume process units

* [Ion implantation]({% link models/prebuilt/ionImplantation.md %}) takes dose in ions/cm² and tilt in degrees. Set the geometry length unit explicitly with `setLengthUnit` (centimeters per unit; `1e-7` for nm). Concentrations are stored per geometry unit cubed by default.
* [Annealing]({% link models/prebuilt/anneal.md %}) takes temperature in kelvin and duration in seconds. Diffusivities use geometry unit²/s, activation energies use eV, and concentration parameters must match the cell-set fields.
* [Thermal oxidation]({% link models/prebuilt/oxidation.md %}) uses degrees Celsius, hours, and micrometers for its temperature, time, and geometry controls.
