---
layout: default
title: Planarize Geometry
parent: Miscellaneous
nav_order: 1
---

# Planarize a Geometry
{: .fs-9 .fw-500}

```c++
#include <psPlanarize.hpp>
```
---

With this class, the user is able to planarize the domain at a specified cutoff position. 
The planarization process involves subtracting a plane from all materials within the domain using a boolean operation.


__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>
```c++
psPlanarize<double, 3>(domain, 
                       0. /*cut off height in z-direction*/).apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>
```python
vps.Planarize(geometry=domain, cutoffHeight=0.).apply()
```
</details>

