---
layout: default
title: Level Set
parent: Geometry Output
nav_order: 1
---

# Level Set
{: .fs-9 .fw-500 }

---

## Saving the Domain

Documentation Coming soon
{: .label .label-yellow}

## Visualization

ViennaPS provides a feature enabling users to save the level set grid points explicitly for each material layer within the domain in the VTK file format. This export includes the level set value associated with each grid point. Users also have the option to specify a width parameter, determining the number of grid points around the zero level set. This functionality enhances the ability to analyze and visualize the level set information in a detailed and customizable manner.

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
domain->saveLevelSetMesh("fileNamePrefix", 3 /* width */);
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
domain.saveLevelSetMesh(fileName="fileNamePrefix", width=3)
```
</details>