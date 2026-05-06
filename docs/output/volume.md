---
layout: default
title: Volume Mesh
parent: Geometry Output
nav_order: 3
---

# Volume Mesh
{: .fs-9 .fw-500 }

---

Volume meshes can be saved using the domain member function `saveVolumeMesh`. The generated `.vtu` mesh is intended for visualization and post-processing, not as input for subsequent ViennaPS simulations.

Hull and disk meshes are also available as VTK `.vtp` outputs. Hull meshes visualize the exterior hull of the Level-Set stack, while disk meshes contain the material-resolved disk representation with material IDs.

## API

```c++
void saveVolumeMesh(const std::string &fileName,
                    double wrappingLayerEpsilon = 1e-2) const;

SmartPointer<viennals::Mesh<NumericType>>
getHullMesh(NumericType bottomExtension = 0.0,
            bool sharpCorners = false) const;

void saveHullMesh(const std::string &fileName,
                  NumericType bottomExtension = 0.0,
                  bool sharpCorners = false) const;

SmartPointer<viennals::Mesh<NumericType>> getDiskMesh() const;

void saveDiskMesh(const std::string &fileName) const;
```

| Parameter | Description |
|-----------|-------------|
| `fileName` | Output file name or prefix passed to the VTK writer. |
| `wrappingLayerEpsilon` | Tolerance used for wrapping layers in the visualization mesh. |
| `bottomExtension` | Extends hull meshes downward by the given distance when positive. |
| `sharpCorners` | Preserve sharp features during hull meshing. |

__Example usage:__

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue }
</summary>
```c++
auto domain = ps::Domain<NumericType, D>::New();
...
// create geometry in domain
...
domain->saveVolumeMesh("fileName");

// Optionally adjust the wrapping tolerance
domain->saveVolumeMesh("fileName_fine", 5e-3);

// Save hull and disk meshes
domain->saveHullMesh("hull.vtp", 5.0, true);
domain->saveDiskMesh("disk.vtp");
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
domain.saveVolumeMesh("fileName")

# Optionally adjust the wrapping tolerance
domain.saveVolumeMesh("fileName_fine", 5e-3)

# Save hull and disk meshes
domain.saveHullMesh("hull.vtp", 5.0, True)
domain.saveDiskMesh("disk.vtp")
```
</details>
