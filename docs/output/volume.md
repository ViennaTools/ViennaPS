---
layout: default
title: Volume Mesh
parent: Geometry Output
nav_order: 3
---

# Volume Mesh
{: .fs-9 .fw-500 }

---

Volume meshes can be saved using the domain member function `saveVolumeMesh`. The mesh is solely for visualization purposes and can not be used for further simulations.

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
...
// create geometry in domain
...
domain->saveVolumeMesh("fileName");
```
</details>

<details markdown="1">
<summary markdown="1">
Python:
{: .label .label-green }
</summary>
```python
domain = vps.Domain()
...
# create geometry in domain
...
domain.saveVolumeMesh(fileName="fileName")
```
</details>