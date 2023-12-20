---
layout: default
title: Simulation Domain
nav_order: 5
has_children: true
---

# Simulation Domain
{: .fs-9 .fw-700}

```c++
#include <psDomain.hpp>
```
---

In our simulation framework, the essential hub for all geometry details is the `psDomain` class. This class is like a smart keeper of information, holding everything about the materials in the simulation domain. It uses level sets to show surfaces and material interfaces with great detail and organizes data in a cell-based structure for the underlying volumes. Depending on the specific process, it can use one or both of these methods. This flexibility ensures that the simulation can handle different processes accurately and efficiently.

