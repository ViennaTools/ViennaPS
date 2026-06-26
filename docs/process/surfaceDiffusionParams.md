---
layout: default
title: Surface Diffusion Parameters
parent: Running a Process
nav_order: 5
---

# Surface Diffusion Parameters
{: .fs-9 .fw-500}

```c++
#include <process/psSurfaceDiffusion.hpp>
```

---

`SurfaceDiffusionParameters` configure the graph-based surface diffusion solver
used by process models that expose surface diffusion coefficients.

{: .note }
Surface diffusion is not a standalone pre-built `ProcessModel`. It is a
process-level capability used by compatible surface models, and it can also be
used directly through `SurfaceDiffusionStencil` and `SurfaceDiffusionSolver`.

## Process Parameters

The parameters configure graph construction for process-integrated surface
diffusion. They are set through the standard `Process::setParameters(...)`
interface.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stabilityFactor` | `double` | `1.0` | Factor used by process-integrated diffusion when choosing stable explicit substeps. |
| `kNeighbors` | `int` | `16` | Number of nearest neighbors considered when `radius <= 0`. |
| `radius` | `double` | `0.0` | Radius-based neighbor search cutoff. `0.0` uses `kNeighbors` instead. |
| `normalCutoff` | `double` | `0.25` | Minimum normal dot product allowed between neighboring points. |
| `sigmaNormal` | `double` | `0.35` | Controls how strongly normal differences reduce diffusion weights. |
| `normalizeByLocalScale` | `bool` | `true` | Scale weights by the local point spacing. |
| `symmetrizeWeights` | `bool` | `true` | Build symmetric graph weights from the directed neighbor graph. |

## Example Usage

<details markdown="1">
<summary markdown="1">
C++
{: .label .label-blue}
</summary>

```c++
using namespace viennaps;

SurfaceDiffusionParameters diffusion;
diffusion.kNeighbors = 16;
diffusion.radius = 0.0;
diffusion.normalCutoff = 0.25;

Process<double, 3> process(domain, model, processTime);
process.setParameters(diffusion);
process.apply();
```
</details>

<details markdown="1">
<summary markdown="1">
Python
{: .label .label-green}
</summary>

```python
import viennaps as vps

diffusion = vps.SurfaceDiffusionParameters()
diffusion.kNeighbors = 16
diffusion.radius = 0.0
diffusion.normalCutoff = 0.25

process = vps.Process(domain, model, process_time)
process.setParameters(diffusion)
process.apply()
```
</details>

## Standalone Solver

ViennaPS also exposes the underlying graph solver directly. It builds a
neighborhood graph from surface point positions and normals, then applies a
graph Laplacian to diffuse scalar data along the surface while reducing coupling
across sharp changes in surface normal.

The checked-in example computes a particle flux on a trench geometry, builds a
`PointCloud` from the flux mesh nodes and normals, and repeatedly applies
`SurfaceDiffusionSolver::stepExplicit(...)` to smooth the `particleFlux` field.

```c++
PointCloud<double> cloud;
cloud.positions = flux->getNodes();
cloud.normals = *flux->getNormals();

SurfaceDiffusionParameters params;
params.kNeighbors = 16;

SurfaceDiffusionSolver<double> solver(
    SurfaceDiffusionStencil<double>(cloud, params));

auto currentFlux = *flux->getCellData().getScalarData("particleFlux");
currentFlux = solver.stepExplicit(currentFlux, 1e-3, 1.0);
```

## Practical Notes

* Use `radius` when a physical cutoff is known; otherwise use `kNeighbors`.
* Increase `kNeighbors` or `radius` if the graph is too disconnected.
* Raise `normalCutoff` to reduce diffusion across corners and steep surface
  changes.
* `SurfaceDiffusionParameters` only affect process-integrated diffusion when
  the selected process model provides surface diffusion coefficients.

## Related Example

* [Surface Diffusion](https://github.com/ViennaTools/ViennaPS/tree/master/examples/surfaceDiffusion)
