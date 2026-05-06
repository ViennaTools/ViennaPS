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

## API

```c++
void saveVolumeMesh(const std::string &fileName,
                    double wrappingLayerEpsilon = 1e-2) const;
```

| Parameter | Description |
|-----------|-------------|
| `fileName` | Output file name or prefix passed to the VTK writer. |
| `wrappingLayerEpsilon` | Tolerance used for wrapping layers in the visualization mesh. |

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
```
</details>
