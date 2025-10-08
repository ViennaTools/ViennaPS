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


In ViennaPS, the central component for managing all geometric information is the **`Domain`** class. It encapsulates the complete material and geometry description of the simulation region. The `Domain` maintains **level sets** to represent surfaces and material interfaces with high accuracy, while also providing a **cell-based data structure** for volume information.

Depending on the applied process, the framework can utilize either or both representations. 
<!-- This design ensures that a wide range of processes—surface-based or volume-based—can be simulated accurately and efficiently within a unified domain structure. -->

