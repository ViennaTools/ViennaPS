# Neutral Transport Cylinder

This example is a neutral-species transport benchmark for a high-aspect-ratio
cylindrical opening. It is intended to compare ViennaPS ray tracing against the
neutral transport model described by Panagopoulos and Lill
(https://doi.org/10.1116/6.0002468), while using the generic ViennaPS geometry
and flux-engine infrastructure.

The example currently runs a diagnostic flux calculation and prints the bottom
transmission probability. The actual level-set advection section is available in
`neutralTransportCylinder.cpp`, but is commented out.

## Geometry

The geometry is generated with `ps::MakeHole` in 3D cylinder mode:

```cpp
ps::MakeHole<NumericType, D>(
    geometry, holeRadius,
    0.0,
    0.0,
    maskHeight, maskTaperAngle,
    ps::HoleShape::QUARTER)
    .apply();
```

This creates:

- a flat Si substrate at the bottom,
- a mask layer above it,
- a cylindrical quarter opening through the mask,
- optional mask taper through `maskTaperAngle`.

The benchmark uses quarter symmetry, so areas in the diagnostic are quarter
circle areas.

## Incoming Flux

The source represents a fixed-pressure Maxwellian gas reservoir above the
feature entrance. The incoming molecular flux density is computed from
molecular effusion:

```text
Gamma_source = p / sqrt(2*pi*m*k_B*T)
```

where:

- `p` is `sourcePressure`,
- `T` is `sourceTemperature`,
- `m` is the particle mass from `sourceMolecularMass`,
- `k_B` is the Boltzmann constant.

The ray tracer returns a dimensionless local transport factor, stored as
`neutralFlux`. Physical fluxes are recovered by multiplying with
`Gamma_source`.

## Ray Tracing

The primary ray-tracing pass emits particles from the source plane above the
geometry. The angular distribution is cosine-like and controlled by
`sourceExponent`.

For each surface hit, the neutral particle records incoming flux in
`neutralFlux`. Reflection/re-emission is handled by the particle model:

- on the etch-front material, sticking is `etchFrontSticking`;
- elsewhere, sticking is coverage dependent:

```text
s = zeroCoverageSticking * (1 - theta)
```

For the paper benchmark, the etch front is Si and is usually configured with
`etchFrontSticking=1.0`.

## Coverage Model

The coverage variable `theta` is the fraction of occupied surface sites by the
neutral etch species.

The model first updates coverage locally from adsorption, desorption, and etch
consumption.

The adsorption coefficient used in the coverage equation is:

```text
a_i = s_i * Gamma_source * neutralFlux_i / (N_A * Gamma_s)
```

where:

- `s_i` is the local sticking coefficient,
- `Gamma_source * neutralFlux_i` is the physical incoming flux,
- `N_A` is Avogadro's number,
- `Gamma_s` is `surfaceSiteDensity` in mol/m^2.

The local coverage loss rate is:

```text
loss_i = k_des,i + k_etch,i
```

with material dependence:

- `k_des,i = desorptionRate` only on `desorptionMaterial` (Mask),
- `k_des,i = 0` elsewhere,
- `k_etch,i = kEtch` only on `etchFrontMaterial`,
- `k_etch,i = 0` elsewhere (Mask).

If `useSteadyStateCoverage=true`, the local reaction part is solved as:

```text
theta_i = a_i / (a_i + loss_i)
```

If `useSteadyStateCoverage=false`, it is updated explicitly:

```text
theta_i <- theta_i + dt * (a_i * (1 - theta_i) - loss_i * theta_i)
```

with `dt = coverageTimeStep`.

Coverage is clamped to `[0, 1]`.

## Desorption Re-Emission

Desorption is implemented as a second ray-tracing pass. After the primary flux
calculation, the surface model computes a desorption source weight per surface
point:

```text
dw_i = k_des * theta_i * Gamma_s * N_A / Gamma_source
```

This is dimensionless because the ray-tracing result is normalized to the
source flux density. Desorption weights are only nonzero on
`desorptionMaterial`.

The transport engines then:

1. run the normal source-plane trace,
2. compute desorption source weights from the current coverage,
3. run a second trace from the surface points or triangle centers,
4. normalize the desorption contribution with the same source-area/ray-count
   normalization,
5. add the desorption flux to the primary flux before the coverage and velocity
   updates.

For disk tracing, the helper `psDesorptionSource.hpp` prepares the surface
source data. CPU disk tracing wraps it in the ViennaRay CPU source interface.
GPU disk tracing passes the same data to ViennaRay's GPU surface-source API.

For triangle tracing, both the CPU and GPU engines map desorption weights to
triangle elements, emit from triangle centers with triangle normals, and apply
the same area correction before normalization. The CPU triangle engine also
records the resulting per-triangle flux for diagnostics.

## Surface Diffusion

Surface diffusion is implemented as a graph-based surface diffusion solver.

After the local coverage update, the model builds or reuses a surface graph
from the current disk-surface point cloud.

The active diffusion points are those whose material matches:

```text
surfaceDiffusionMaterial (Mask)
```

Two points are connected by a diffusion edge if they are within
`surfaceDiffusionNeighborDistance`. If this value is `0.0`, the code estimates
a cutoff from the current surface point spacing using a KD-tree. The current
estimate is approximately:

```text
1.75 * average nearest-neighbor distance
```

with a small safeguard based on the minimum positive spacing.

Each edge has weight:

```text
w_ij = 1 / |x_i - x_j|^2
```

The implicit graph diffusion equation is:

```text
theta_new - D_s * dt * L_graph(theta_new) = theta_after_reactions
```

Equivalently, for active point `i`:

```text
theta_i_new
+ D_s * dt * sum_j w_ij * (theta_i_new - theta_j_new)
= theta_i_after_reactions
```

where `j` are active neighboring surface points.

The linear system is solved with conjugate gradient and a diagonal Jacobi
preconditioner:

```text
diag_i = 1 + D_s * dt * sum_j w_ij
```

The solve is controlled by:

- `surfaceDiffusionCoefficient`,
- `surfaceDiffusionNeighborDistance`,
- `surfaceDiffusionSolverTolerance`,
- `surfaceDiffusionMaxIterations`.

The graph naturally gives zero-flux boundaries where the active material region
ends or where there are no active neighboring points.

Diffusion is currently operator-split after the local reaction
coverage update. With `useSteadyStateCoverage=true`, the local reaction part is
made steady state first and diffusion is applied afterward; this is not a
single fully coupled steady-state reaction-diffusion solve.

## Etch Velocity

Etch velocity is only applied on `etchFrontMaterial`, currently Si. The velocity
uses the local neutral coverage:

```text
v_etch = kEtch * Gamma_s * theta / rho_Si
```

where:

- `Gamma_s` is `surfaceSiteDensity`,
- `rho_Si` is `siliconDensity`.

The velocity is converted into the configured ViennaPS length/time units and
given a negative sign for etching.

## Transmission Probability Diagnostic

The example prints a bottom transmission probability after
`diagnosticProcess.calculateFlux()`.

For CPU and GPU triangle tracing, the triangle mesh stores per-triangle
`neutralFlux`.
The diagnostic identifies horizontal bottom etch-front triangles inside
`holeRadius`, integrates `neutralFlux * area`, and divides by the analytic
quarter-circle source aperture area:

```text
alpha = integral_bottom(neutralFlux dA) / A_top
```

For this quarter-cylinder geometry:

```text
A_top = pi * topRadius^2 / 4
```

where:

```text
topRadius = holeRadius + tan(maskTaperAngle) * maskHeight
```

Because `neutralFlux` is normalized to the imposed source flux density, the top
source integral is represented by the aperture area.

The diagnostic currently requires a triangle flux engine because it reads the
per-triangle flux mesh via `Process::getTriangleMesh()`. Use `CT` for CPU
triangle tracing or `GT` for GPU triangle tracing.

## Main Configuration Parameters

Geometry:

- `gridDelta`: grid spacing in the configured length unit.
- `xExtent`, `yExtent`: lateral domain extent.
- `holeRadius`: bottom/nominal hole radius.
- `maskHeight`: mask thickness.
- `maskTaperAngle`: mask sidewall taper angle in degrees.

Ray tracing:

- `fluxEngine`: use `CT` for CPU triangle tracing or `GT` for GPU triangle
  tracing and the transmission diagnostic.
- `raysPerPoint`: number of rays per source point.
- `sourceExponent`: cosine-distribution exponent.

Reservoir source:

- `sourcePressure`
- `sourceTemperature`
- `sourceMolecularMass`

Coverage and reaction:

- `zeroCoverageSticking`
- `etchFrontSticking`
- `desorptionRate`
- `desorptionMaterial`
- `kEtch`
- `surfaceSiteDensity`
- `siliconDensity`
- `coverageTimeStep`
- `useSteadyStateCoverage`

Surface diffusion:

- `surfaceDiffusionCoefficient`
- `surfaceDiffusionMaterial`
- `surfaceDiffusionNeighborDistance`
- `surfaceDiffusionSolverTolerance`
- `surfaceDiffusionMaxIterations`

Coverage initialization:

- `coverageInitIterations`
- `coverageTolerance`

## Build And Run

The commands below assume you are in the ViennaPS repository root.

```bash
cd /path/to/ViennaPS
```

### Flux Engine Choice

The bottom transmission diagnostic requires a triangle flux engine because it
reads per-triangle flux data from `Process::getTriangleMesh()`.

Use one of:

```text
fluxEngine=CT  # CPU triangle engine
fluxEngine=GT  # GPU triangle engine
```

The checked-in `config.txt` currently uses `GT`. For CPU-only builds, either
edit `config.txt` to use `CT` or use the temporary substitution shown below.

By default, ViennaPS fetches ViennaRay from:

```text
feature/gpu-surface-source
```

Local sibling ViennaRay checkouts are opt-in through
`VIENNAPS_USE_LOCAL_VIENNARAY=ON`.

### Available Presets

To list all configure and build presets:

```bash
cmake --list-presets=all
```

This repository currently provides:

- configure preset `cpu`, building into `build-cpu` and fetching the required
  ViennaRay branch;
- build preset `neutral-transport-cylinder-cpu`;
- configure preset `gpu-viennaray-branch`, building into `build`;
- build preset `neutral-transport-cylinder`.

### CPU With Presets

Configure from scratch:

```bash
rm -rf build-cpu
cmake --preset cpu
```

Build this example:

```bash
cmake --build --preset neutral-transport-cylinder-cpu
```

Run with CPU triangle diagnostics without editing `config.txt`:

```bash
cd build-cpu/examples/neutralTransportCylinder
./neutralTransportCylinder <(sed 's/fluxEngine=GT/fluxEngine=CT/' config.txt)
```

Alternatively, set this in `config.txt` before running:

```text
fluxEngine=CT
```

### GPU With Presets

The `gpu-viennaray-branch` preset enables GPU support, builds examples, uses
`gcc-12/g++-12`, disables the local ViennaRay override, and fetches ViennaRay
from the branch:

```text
feature/gpu-surface-source
```

Configure from scratch:

```bash
rm -rf build
cmake --preset gpu-viennaray-branch
```

Build this example:

```bash
cmake --build --preset neutral-transport-cylinder
```

Run with the default GPU triangle config:

```bash
cd build/examples/neutralTransportCylinder
./neutralTransportCylinder config.txt
```

### CPU Without Presets

Configure from scratch without GPU support:

```bash
rm -rf build-cpu
cmake -S . -B build-cpu \
  -DVIENNAPS_USE_GPU=OFF \
  -DVIENNAPS_BUILD_EXAMPLES=ON \
  -DVIENNAPS_USE_LOCAL_VIENNARAY=OFF \
  -DVIENNAPS_VIENNARAY_BRANCH=feature/gpu-surface-source
```

Build only this example:

```bash
cmake --build build-cpu --target neutralTransportCylinder -j 4
```

Run with CPU triangle diagnostics:

```bash
cd build-cpu/examples/neutralTransportCylinder
./neutralTransportCylinder <(sed 's/fluxEngine=GT/fluxEngine=CT/' config.txt)
```

### GPU Without Presets

For GPU support, use a CUDA-supported host compiler. With CUDA 12.0, GCC 12 is
safe; GCC 13 triggers NVCC's unsupported compiler check.

Configure from scratch with GPU support and the required ViennaRay branch:

```bash
rm -rf build
cmake -S . -B build \
  -DVIENNAPS_USE_GPU=ON \
  -DVIENNAPS_BUILD_EXAMPLES=ON \
  -DVIENNAPS_USE_LOCAL_VIENNARAY=OFF \
  -DVIENNAPS_VIENNARAY_BRANCH=feature/gpu-surface-source \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12
```

Build the example:

```bash
cmake --build build --target neutralTransportCylinder -j 4
```

Run with the default GPU triangle flux engine:

```bash
cd build/examples/neutralTransportCylinder
./neutralTransportCylinder config.txt
```

## Current Benchmark Scope

This example currently implements:

- fixed-pressure molecular effusion input flux,
- ballistic ray-traced Knudsen transport,
- coverage-dependent sticking,
- material-dependent desorption sink and desorption source emission,
- secondary ray tracing of desorbed species,
- material-dependent etch-front consumption,
- coverage-dependent etch velocity,
- graph-based surface diffusion on the selected material,
- coverage initialization iterations before the diagnostic flux calculation,
- CPU and GPU triangle-engine bottom transmission probability diagnostics.

The current diagnostic path is fixed-geometry: it calls
`diagnosticProcess.calculateFlux()` and prints transmission. The level-set
advection/etching part is present in the source file but commented out, so this
example does not currently advance the geometry unless that block is enabled.

Current approximations relative to the paper:

- the geometry is a generic ViennaPS level-set quarter-cylinder, not the
  paper's dedicated 2D-axisymmetric angular-coefficient discretization;
- surface diffusion is a graph diffusion solve on disk-surface points, not an
  area-weighted finite-volume or cotangent Laplace-Beltrami solve on triangles;
- coverage/reaction and surface diffusion are operator-split;
- with `useSteadyStateCoverage=true`, the local reaction part is made
  steady-state first and diffusion is applied afterward, so this is not a fully
  coupled steady-state reaction-diffusion solve;
- desorption is material dependent and currently configured for the mask, while
  etch consumption is configured for the Si etch front;
