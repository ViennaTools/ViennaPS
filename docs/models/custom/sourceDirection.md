---
layout: default
title: Primary Source Direction
parent: Custom Models
grand_parent: Process Models
nav_order: 4
---

# Primary Source Direction
{: .fs-9 .fw-500}

---

The primary source direction is an optional parameter that allows users to tailor the initial particle direction distribution by specifying a tilt during flux calculation from a source plane. In cases where no primary source direction is explicitly defined, it defaults to being aligned with the surface normal of the source plane.

{: .note}
> If there is no intention to tilt the initial distribution, it is advisable not to set the primary source direction equal to the source plane normal. Instead, using the default value is recommended for a slight performance advantage.

__Example usage:__

```c++
auto myModel = psSmartPointer<psProcessModel<NumericType, D>>::New();
double tiltingAngle = 30. * M_PI / 180.; // tilting angle of 30 degree
double x = -std::sin(tiltingAngle);
double y = -std::cos(tiltingAngle);
myModel->setPrimaryDirection({x, y, -1.});
```