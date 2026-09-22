---
layout: default
title: General Notes
nav_order: 4
---

# General Notes
{: .fs-9 .fw-700 .no_toc}

---

These notes cover the C++ and Python interfaces and practical guidelines for
setting up simulations. Numerical recommendations are starting points: check
convergence for the geometry, process model, and quantities you want to measure.

## Contents
{: .no_toc }

- TOC
{:toc}

## Switching between 2D and 3D

In C++, the simulation dimension is selected at compile time, usually as the
second template parameter:

```cpp
viennaps::Domain<double, 3> domain3D;
viennaps::Domain<double, 2> domain2D;
```

Use the same dimension for the domains, models, and processes in a simulation.
The [Extrude utility]({% link misc/extrusion.md %}) can create a 3D domain from a
2D domain when you want to continue with an extruded geometry.

Python defaults to 2D. Select the dimension before creating domains, geometries,
or models:

```python
import viennaps as ps

ps.setDimension(3)  # Use 2 for a 2D simulation.
```

The dimension-specific modules are also available explicitly:

```python
ps.d2.Domain  # 2D domain class
ps.d3.Domain  # 3D domain class
```

Changing the default with `setDimension` does not convert existing objects.
In the standard domain setup, the vertical process direction is **y in 2D and z
in 3D**; the `yExtent` argument is ignored in 2D. `MakeHole` in 2D creates a trench;
it is not an axisymmetric simulation of a cylindrical hole.

## Namespace

C++ classes and utilities are in the `viennaps` namespace. Qualify their names or
use a namespace alias:

```cpp
namespace ps = viennaps;
```

The documentation sometimes omits the namespace for brevity. Python examples on
this page use `import viennaps as ps`.

## Numeric Types (C++ only)

Use `double` for level-set calculations unless you have a specific reason to use
`float`. The numeric type is a compile-time template parameter and remains fixed
for each instantiated object. Use compatible types for the components of a
simulation. The Python bindings use `double`.

Ray-tracing geometry still uses `float`, even when the level sets use `double`.
Choose suitable coordinate scales as described below.

## Using Smart Pointers (C++ only)

Domains and process models are passed between components through `SmartPointer`,
ViennaPS's shared-pointer wrapper. Construct them with `New`:

```cpp
using namespace viennaps;
using NumericType = double;
constexpr int D = 3;

auto domain = SmartPointer<Domain<NumericType, D>>::New();
// Equivalent shorter form: Domain<NumericType, D>::New()

auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(-0.1);
```

Sharing a pointer shares the underlying object. Use a domain copy when independent
geometry is required; see [Process state and configuration](#process-state-and-configuration).

## Numerical length scale

**Grid spacing and characteristic geometry sizes should not be much smaller than
`1e-4` in the coordinates supplied to the library.** Prefer a comfortable margin
above this scale for ray-traced processes. Ray-tracing geometry uses `float`, even
when the level sets use `double`, so very small features can suffer from numerical
precision errors. Several geometry builders also use absolute offsets of `1e-4`.
Using `double` for the simulation does not remove these limits.

For example, express a 100 nm feature with a 10 nm grid as `0.1` and `0.01` in
micrometers, or `100` and `10` in nanometers. Supplying the same lengths as `1e-7`
and `1e-8` in meters is a poor choice for ray tracing with the default settings.
Keep the geometry near the coordinate origin as well: large coordinate offsets
relative to the feature size also reduce the precision available to `float`.

The `1e-4` recommendation is a practical scale guideline, not a universal hard
lower bound imposed by floating-point arithmetic. There is also a concrete
ray-intersection constraint: `RayTracingParameters.minRayDistance` defaults to
`1e-4`, and the CPU engines require it to be **less than `gridDelta / 2`** before
forwarding it to the tracer. Thus, their default passes this check only for
`gridDelta > 2e-4`. Otherwise they warn that surface hits may be missed. Rescale
the problem first; reducing the ray distance alone does not fix float precision
or the geometry builders' offsets. Parameter support also differs by engine:
the current GPU triangle engine does not forward `minRayDistance`.

## Units and model conventions

Set length and time units before constructing physical models that require them:

```python
ps.Length.setUnit("um")
ps.Time.setUnit("s")
```

These setters select conversion conventions; they do not rescale an existing
domain, grid, or arbitrary user-supplied rates. When changing the coordinate unit,
convert every geometry length, grid spacing, thickness, and applicable rate
consistently. For a simple velocity model, rates are in your chosen length per
time and durations must use the matching time unit. Unit settings are global,
so keep them consistent across simulations in the same process.

Check each physical model's parameter conventions separately. Plasma models
convert their internally calculated rates to the selected units, but energies,
fluxes, and reaction constants retain the conventions documented in that model.
`Oxidation`, for example, uses micrometers, hours, degrees Celsius, and atmospheres.
Angles also need individual attention: geometry taper angles and IBE tilt angles
use degrees, while SF6/O2 ion reflection parameters use radians internally.

See [Units]({% link misc/units.md %}) for supported units and volume-process conventions.

## Grid resolution and convergence

Choose `gridDelta` relative to the smallest opening, film, or curvature that
matters to the result. A feature comparable to one grid cell is poorly resolved.
Several cells across the smallest important feature are a useful starting point;
there is no universal cell count that guarantees accuracy.

Repeat a representative simulation with a finer grid, such as half the spacing,
and compare quantities such as etch depth, sidewall angle, remaining mask
thickness, or deposited thickness. Check ray statistics and time stepping
separately so Monte Carlo noise does not obscure the grid comparison. For models
with volume solvers, halving the spacing increases the number of cells in a fixed
volume by roughly 4 times in 2D or 8 times in 3D; runtime can grow further because
of solver work and smaller time steps.

## Boundaries and symmetry

The convenience domain constructor uses reflective lateral boundaries and an
infinite vertical boundary. Choose periodic boundaries for a repeating cell and
reflective boundaries only where mirror symmetry is appropriate. Ray tracing
normally follows the domain boundaries; `ignoreFluxBoundaries=True` changes
particle transport and should be a deliberate modeling choice.

Use half or quarter geometries only when both geometry and incident flux have
the required symmetry. A tilted beam can break that symmetry. Half/quarter hole
helpers do not support periodic boundaries.

See [Domain Setup]({% link domain/setup.md %}) for boundary configuration.

## Material layers and wrapping

Use `insertNextLevelSetAsMaterial` to keep level sets and material assignments
together. By default, each newly inserted level set is unioned with the previous
one, so the last level set represents the complete exposed surface. Insertion
order is therefore part of the geometry definition; it is not simply a list of
independent material solids. The custom-process example inserts the mask before
the substrate so that the final surface includes both.

Construct all level sets on a consistent grid with matching spacing and boundary
conditions. Disable `wrapLowerLevelSet` only when you have already constructed
the required nested representation. Before depositing a distinct material, use
`domain.duplicateTopLevelSet(material)` to preserve the underlying material
interface and give the new film its own level set.

See [Surface and Material Interfaces]({% link domain/surface.md %}) and
[Material Mapping]({% link domain/material.md %}) for the domain representation.

## Rate signs and masking

For `IsotropicProcess` and `SingleParticleProcess`, positive rates grow material
and negative rates remove it. Physical etching models can instead accept positive
physical inputs and calculate negative surface velocities internally. Do not
transfer a sign convention from one model to another without checking it.

A material named `Mask` is not automatically immobile in every model. Configure
`maskMaterial`/`maskMaterials` or a zero material rate where appropriate. The
simple-process constructors default to `Material.Undefined` as their masking
material; the SF6/O2 model explicitly includes mask sputtering. Verify mask
erosion and material selectivity on a small example before running a full recipe.

## Choosing a process model

Use geometric distributions for prescribed shape changes, analytic velocity
models for prescribed rates, and particle models when visibility, angular flux,
reflection, or transport must determine the result. A geometrically deposited
thickness and a particle deposition duration are different inputs.

For example, `SphereDistribution(radius=layerThickness)` applies a geometric
offset; the trench example executes it with a process duration of zero. A
time-dependent `SingleParticleProcess` needs an explicit positive duration.
Default physical parameters and example configurations are starting points for
calibration, not a guarantee that a specific fabrication recipe is reproduced.

See [Pre-Built Models]({% link models/prebuilt/index.md %}) for available models.

## Ray statistics and smoothing

`RayTracingParameters.raysPerPoint` defaults to `1000`; treat this as a starting
point. Increase it until the measured result is sufficiently stable, especially
in shadowed regions and deep features. Refine the grid and increase ray counts
as separate checks.

Keep smoothing modest and test its effect on narrow features. The default
`smoothingNeighbors` is `1`. In the CPU triangle engine, the flux conversion
radius is `gridDelta * (smoothingNeighbors + 1)`, so the same setting has a
different physical size after grid refinement. Heavy smoothing can conceal
spatial variation without establishing statistical convergence. Likewise,
reducing `maxReflections` or `maxBoundaryHits` can change transport; check the
result if using these limits to reduce runtime.

See [Ray Tracing Parameters]({% link process/rayTracingParams.md %}) for the available controls.

## Reproducibility and flux engines

For repeatable debugging, set **both** `useRandomSeeds=False` and `rngSeed`.
Setting the seed alone leaves random seeding enabled. Repeat with other seeds
when estimating sensitivity to Monte Carlo noise.

Select the flux engine explicitly when comparing runs across machines. `AUTO`
can choose a GPU engine when a CUDA device and a GPU implementation of the model
are available; otherwise the normal CPU choice is `CPU_DISK`. The current
GPU auto-selection chooses disks for periodic domains and triangles otherwise.
`GPU_LINE` is for 2D and falls back to GPU triangles in 3D.

Record the library version, engine, dimension, thread count, seed, and numerical
parameters. A fixed seed is not a promise of identical results across different
engines, versions, or execution configurations. CPU/GPU comparisons should use
appropriate numerical tolerances.

## Advection and time stepping

For ordinary level-set advection, use `0 < timeStepRatio < 0.5`; the default is
`0.4999`. This is a CFL ratio controlling surface motion relative to the grid,
not a duration in seconds. Reduce it when checking temporal convergence or
investigating unstable or sensitive evolution. The process duration controls the
total simulated time, while the solver chooses internal time steps.

Use `spatialScheme` and `temporalScheme` explicitly when comparing numerical
schemes; `integrationScheme` is deprecated. For Runge-Kutta integration of a
flux-dependent process, consider `calculateIntermediateVelocities=True` to
recompute fluxes and velocities on intermediate surfaces. It adds ray-tracing
work; choosing a higher-order temporal scheme alone does not enable these
recalculations. Models with their own solvers, such as oxidation, have additional
time-step controls.

See [Advection Parameters]({% link process/advectionParams.md %}) for scheme and time-step settings.

## Coverage initialization

For models with surface coverages, choose a positive
`CoverageParameters.tolerance` and a finite `maxIterations`, then inspect the
convergence messages. If both are left at their defaults, the flux strategy uses
10 initialization iterations. Setting only a positive tolerance leaves a very
large iteration limit, which is undesirable if sampling noise prevents
convergence.

The convergence metric is the mean **squared** change per coverage field. A
tolerance of `1e-4` therefore corresponds to an RMS change of `1e-2`, not a maximum
pointwise change of `1e-4`. Reaching the iteration cap does not prove convergence:
the current implementation marks initialization complete even when the cap is
reached. Adjust ray statistics and tolerance together, and avoid marking
coverages `initialized=True` just to skip this step on a fresh geometry.

See [Coverage Parameters]({% link process/coverageParams.md %}) for initialization controls.

## Process state and configuration

`Process.apply()` evolves the supplied domain in place. Running another process
on it continues from the changed geometry. For independent parameter studies,
rebuild the initial geometry or use `ps.Domain(original)` to copy level sets and
material assignments. For simulations with cell data, note that the current
domain copy reconstructs the cell set rather than copying all stored field
values.

Geometry builders such as `MakeTrench` and `MakeHole` clear existing layers when
applied. Use insertion, duplication, and Boolean operations to extend an existing
structure. Also, `Process.setParameters(...)` copies its parameter object: after
editing that object, call `setParameters` again to apply the change.

Python `readConfigFile` returns numeric values as floats, including integer-like
entries. Cast counts such as rays and cycles to `int`. A configuration value only
affects the simulation when the script passes it to the appropriate API.

## Inspecting and saving results

Save the initial geometry and inspect material IDs, interfaces, openings, and
boundary placement. Use `saveSurfaceMesh(..., addInterfaces=True)` when material
interfaces matter. For a particle model, `process.calculateFlux()` lets you
inspect fluxes on the current geometry without advancing the surface. A flat
wafer or simple trench is a useful check of rate magnitude, sign, and uniformity.
Keep logging at `INFO` while validating a setup so initialization and termination
messages remain visible.

Use VTK meshes for visualization and `Writer`/`Reader` with `.vpsd` for saving and
restoring level-set domains and material assignments. A `.vpsd` file is not a
complete process checkpoint: current `Writer` code does not serialize cell-set
contents or the process/model configuration. Save the recipe and unit settings
alongside it. `enableMetaData(MetaDataLevel.FULL)` can add numerical parameters to
mesh outputs, but does not replace a complete run record.

See [Geometry Output]({% link output/index.md %}),
[VTK Metadata Export]({% link output/metadata.md %}), and
[Logging]({% link misc/logging.md %}) for output options.

## Example: explicit settings for a small deposition run

This example uses micrometers and seconds, preserves the deposited material as a
separate layer, and fixes the engine and seed for debugging. The grid and ray
count still need convergence checks for a production result.

```python
import viennaps as ps

ps.setDimension(2)
ps.Length.setUnit("um")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.INFO)

domain = ps.Domain(gridDelta=0.05, xExtent=4.0)
ps.MakeTrench(domain, trenchWidth=1.0, trenchDepth=1.0).apply()
domain.duplicateTopLevelSet(ps.Material.SiO2)
domain.enableMetaData(ps.MetaDataLevel.FULL)

model = ps.SingleParticleProcess(
    rate=0.1, stickingProbability=0.5, sourceExponent=1.0
)
process = ps.Process(domain, model, 1.0)
process.setFluxEngineType(ps.FluxEngineType.CPU_DISK)

rays = ps.RayTracingParameters()
rays.raysPerPoint = 1000
rays.useRandomSeeds = False
rays.rngSeed = 42
process.setParameters(rays)

advection = ps.AdvectionParameters()
advection.timeStepRatio = 0.25
process.setParameters(advection)

domain.saveSurfaceMesh("initial.vtp", addInterfaces=True)
process.apply()
domain.saveSurfaceMesh("final.vtp", addInterfaces=True)
```
