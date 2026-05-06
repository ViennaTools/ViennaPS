---
layout: default
title: Surface Mesh
parent: Geometry Output
nav_order: 2
---

# Surface Mesh
{: .fs-9 .fw-500 }

---

> Surface mesh export can write either the top visible surface or all material interfaces.
> Material IDs are stored in the mesh point data under the `"MaterialIds"` label.


## Overview

Use the domain member function `saveSurfaceMesh` or `getSurfaceMesh` to export the triangulated surface of a geometry.  
The function can output either:

- the **topmost surface** (default), or  
- **all material interfaces** when `addInterfaces` is enabled.

---

## API

### Get the surface mesh

```c++
SmartPointer<viennals::Mesh<NumericType>>
getSurfaceMesh(bool addInterfaces = false,
               bool sharpCorners = false,
               double minNodeDistanceFactor = 0.01) const;
```

* **`addInterfaces`** — include all internal material interfaces (not only top surface).
* **`sharpCorners`** — preserve sharp features during meshing.
* **`minNodeDistanceFactor`** — minimum node distance factor used when `addInterfaces` is enabled.

### Save the surface mesh

```c++
void saveSurfaceMesh(const std::string &fileName,
                     bool addInterfaces = true,
                     bool sharpCorners = false,
                     double minNodeDistanceFactor = 0.01) const;
```

Writes the resulting triangulated mesh to a `.vtp` file including optional material and metadata information.

---

## Example usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>

```c++
auto domain = ps::Domain<double, 3>::New();
// ... create geometry in domain ...

// Save top-level surface only
domain->saveSurfaceMesh("surface.vtp", false);

// Save all material interfaces
domain->saveSurfaceMesh("interfaces.vtp", true);

// Save all interfaces and preserve sharp corners
domain->saveSurfaceMesh("interfaces_sharp.vtp", true, true);
```

</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green }
</summary>

```python
import viennaps as vps

domain = vps.Domain()
# ... create geometry in domain ...

# Save top surface only
domain.saveSurfaceMesh("surface.vtp", False)

# Save all material interfaces
domain.saveSurfaceMesh("interfaces.vtp", True)

# Save all interfaces and preserve sharp corners
domain.saveSurfaceMesh("interfaces_sharp.vtp", True, True)
```

</details>

---

## Returned mesh

When using `getSurfaceMesh()`, the returned object is a `viennals::Mesh` that can be further processed or visualized:

```c++
auto mesh = domain->getSurfaceMesh(true);
viennals::VTKWriter<double> writer(mesh, "mesh.vtp");
writer.apply();
```

---

## Behavior summary

| Option                 | Description                                                    | Default |
| ---------------------- | -------------------------------------------------------------- | ------- |
| `addInterfaces`        | Export all material interfaces instead of only the top surface | `true` for `saveSurfaceMesh`, `false` for `getSurfaceMesh` |
| `sharpCorners`         | Preserve sharp features during meshing                         | `false` |
| `minNodeDistanceFactor` | Minimum node distance factor for multi-surface interface meshing | `0.01` |

---

## Notes

* All exported meshes are in **VTK (.vtp)** format.
* The mesh includes material IDs and metadata when a `MaterialMap` is present in the domain.
* Use `getSurfaceMesh()` for programmatic access or `saveSurfaceMesh()` for direct file output.
