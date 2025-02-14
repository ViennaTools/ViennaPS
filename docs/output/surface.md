---
layout: default
title: Surface Mesh
parent: Geometry Output
nav_order: 2
---

# Surface Mesh
{: .fs-9 .fw-500 }

---

To save a triangulated mesh of the geometry surface, users can use the domain member function `saveSurfaceMesh`. The surface mesh also contains material information is the optional parameter `addMaterialIds` is set to `true`.

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
domain->saveSurfaceMesh("fileName", true);
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
domain.saveSurfaceMesh(fileName="fileName", addMaterialIds=True)
```
</details>