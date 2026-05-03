# Neutral Transport Quarter Cylinder

This example is a neutral-species transport benchmark for a high-aspect-ratio
cylindrical opening. It is intended to compare ViennaPS ray tracing against the
neutral transport model described by Panagopoulos and Lill (https://doi.org/10.1116/6.0002468), while
using the generic ViennaPS geometry and flux-engine infrastructure.

The example currently runs a diagnostic flux calculation and prints the bottom
transmission probability. The actual level-set advection section is available in
`neutralTransportQuarterCylinder.cpp`, but is commented out.

## Geometry

The geometry is generated with `ps::MakeHole` in 3D quarter-cylinder mode:

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

For GPU triangle tracing, desorption weights are computed on the disk mesh,
mapped to triangle elements, and emitted from triangle centers with triangle
normals.

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

For GPU triangle tracing, the triangle mesh stores per-triangle `neutralFlux`.
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

The diagnostic currently requires the GPU triangle engine because it reads the
per-triangle flux mesh via `Process::getTriangleMesh()`.

## Main Configuration Parameters

Geometry:

- `gridDelta`: grid spacing in the configured length unit.
- `xExtent`, `yExtent`: lateral domain extent.
- `holeRadius`: bottom/nominal hole radius.
- `maskHeight`: mask thickness.
- `maskTaperAngle`: mask sidewall taper angle in degrees.

Ray tracing:

- `fluxEngine`: use `GT` for GPU triangle tracing and the transmission
  diagnostic.
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

From the repository root:

```bash
cmake --build build --target neutralTransportQuarterCylinder -j 4
```

From the build output directory, run:

```bash
./examples/neutralTransportQuarterCylinder/neutralTransportQuarterCylinder \
  ../../examples/neutralTransportQuarterCylinder/config.txt
```

Or from this example directory, pass `config.txt` explicitly to the built
executable.

## Current Benchmark Scope

This example currently covers:

- fixed-pressure molecular effusion input flux,
- ballistic ray-traced Knudsen transport,
- coverage-dependent sticking,
- material-dependent desorption sink and desorption source,
- surface re-emission of desorbed species,
- material-dependent etch consumption,
- coverage-dependent etch velocity,
- graph-based surface diffusion on the selected material,
- bottom transmission probability diagnostic.
