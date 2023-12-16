---
layout: default
title: Process Models
nav_order: 7
has_children: true
---

# Process Models
{: .fs-9 .fw-700}


```c++
#include <psProcessModel.hpp>
```
---

All the information about the process is encompassed in the class `psProcessModel`, as it includes all the particle type information required for ray tracing, the surface model, as well as advection callbacks, for generating volume models describing chemical processes inside the material.

Users have the flexibility to configure their own custom process model or opt for pre-defined models encompassing frequently used processes.