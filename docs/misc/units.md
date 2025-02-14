---
layout: default
title: Units
parent: Miscellaneous
nav_order: 6
---

# Units
{: .fs-9 .fw-500}

---

Physical models, like the SF6O2 or Fluorocarbon etching models, require the user to specify the units of the input parameters. The user must set the length and time units before creating a model using the `units` module. The units are global parameters and apply to every model and process in the program.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
using namespace viennaps;
units::setLengthUnits(units::LengthUnit::NANOMETER)
units::setTimeUnits(units::TimeUnit::MINUTE)

// the units can also be specified using strings
units::setLengthUnits("nm") // or "nanometer"
units::setTimeUnits("min") // or "minute"
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>
```python
# in Python strings have to be used to set the units
vps.setLengthUnits("nm") # or "nanometer"
vps.setTimeUnits("min") # or "minute"
```
</details>