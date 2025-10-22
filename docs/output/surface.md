---
layout: default
title: Surface Mesh
parent: Geometry Output
nav_order: 2
---

# Surface Mesh
{: .fs-9 .fw-500 }

---

> From version **4.0.0**, surface mesh export has been extended and renamed for clarity.  
> The old option `addMaterialIds` has been replaced with **`addInterfaces`**.  
> Additional options control how multiple materials and level sets are combined.


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
               double wrappingLayerEpsilon = 0.01,
               bool boolMaterials = false) const;
````

* **`addInterfaces`** — include all internal material interfaces (not only top surface).
* **`wrappingLayerEpsilon`** — small offset used to generate interface wrapping layers (default: 0.01).
* **`boolMaterials`** — perform Boolean subtraction between successive level sets.

### Save the surface mesh

```c++
void saveSurfaceMesh(std::string fileName,
                     bool addInterfaces = true,
                     double wrappingLayerEpsilon = 0.01,
                     bool boolMaterials = false) const;
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
auto domain = ps::SmartPointer<ps::Domain<double, 3>>::New();
// ... create geometry in domain ...

// Save top-level surface only
domain->saveSurfaceMesh("surface.vtp", false);

// Save all material interfaces
domain->saveSurfaceMesh("interfaces.vtp", true);

// Save with additional Boolean-based material separation
domain->saveSurfaceMesh("interfaces_bool.vtp", true, 0.01, true);
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
domain.saveSurfaceMesh(fileName="surface.vtp", addInterfaces=False)

# Save all material interfaces
domain.saveSurfaceMesh(fileName="interfaces.vtp", addInterfaces=True)

# Save with Boolean material separation
domain.saveSurfaceMesh(fileName="interfaces_bool.vtp",
                       addInterfaces=True,
                       wrappingLayerEpsilon=0.01,
                       boolMaterials=True)
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
| `addInterfaces`        | Export all material interfaces instead of only the top surface | `true`  |
| `wrappingLayerEpsilon` | Distance offset for interface wrapping                         | `0.01`  |
| `boolMaterials`        | Apply Boolean subtraction to separate overlapping level sets   | `false` |

---

## Notes

* All exported meshes are in **VTK (.vtp)** format.
* The mesh includes material IDs and metadata when a `MaterialMap` is present in the domain.
* Use `getSurfaceMesh()` for programmatic access or `saveSurfaceMesh()` for direct file output.

