---
layout: default
title: Surface Mesh
parent: Geometry Output
nav_order: 2
---

# Surface Mesh
{: .fs-9 .fw-500 }

---

Documentation Coming soon
{: .label .label-yellow}

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
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