# finOxidation

Simulates thermal oxidation of a silicon fin structure. A rectangular Si fin is
centered at x = 0 on a flat Si substrate; oxide grows simultaneously on the fin top,
both sidewalls, and the surrounding substrate.

## Geometry

```
       │← finWidth →│
   ────┤            ├────  y = finHeight
       │    Si fin  │
       │            │
 ──────┴────────────┴──────  y = 0  (substrate top)
         Si substrate
```

The REFLECTIVE boundary at x = 0 mirrors the domain, so the simulation represents
a half-fin; the output meshes show the full symmetric structure. In 3D the fin
is extruded uniformly along Z (uniform cross-section; no Z variation in geometry or
oxidation rate).

## Configuration Parameters

All lengths are in **micrometers (µm)**, time in **hours (hr)**, pressure in **atm**.

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | `2` | Simulation dimensionality: `2` or `3` |
| `numThreads` | `16` | OpenMP thread count |
| `gridDelta` | `0.01` | Cartesian grid spacing (µm) |
| `xExtent` | `0.6` | Half-width of the domain in X (µm) |
| `zExtent` | *(= xExtent)* | 3D only: half-depth in Z (µm) |
| `finWidth` | `0.2` | Width of the Si fin (µm) |
| `finHeight` | `0.5` | Height of the Si fin above the substrate (µm) |
| `oxideThickness` | `0.0` | Oxide on the silicon at t = 0 (µm); 0 = bare silicon. Thinner than three grid cells, it first grows by the 1-D Deal-Grove law (time charged) until the solver resolves it |
| `oxidationTime` | `0.2` | Total oxidation time (hr) |
| `timeStep` | `0.025` | Maximum internal time step (hr); actual steps are CFL-limited |
| `temperature` | `1000.` | Furnace temperature (°C); valid range 700–1200 °C |
| `pressure` | `1.` | Ambient pressure (atm); scales both B and B/A linearly |
| `oxidant` | `wet` | `wet` (H₂O) or `dry` (O₂) |
| `orientation` | `100` | Crystal orientation: `100`, `110`, `111`, or `poly` |
| `maxGridPoints` | *(5M)* | Optional cap on Cartesian solve grid nodes |
| `useGpu` | `cpu` | Large linear solves on the GPU: `cpu`, `gpu` or `auto` (GPU requires `VIENNALS_USE_GPU=ON`) |
| `outputPrefix` | `ps_fin_oxidation` | Prefix for all output file names |

### Grid Sizing Guide

The diffusion solve scales as O((L/δ)^D): halving `gridDelta` increases solve
time ~4× in 2D and ~8× in 3D.

| Scenario | `gridDelta` |
|---|---|
| 2D quick test | 0.05 µm |
| 2D production | 0.01 µm |
| 3D quick test | 0.1 µm |
| 3D production | 0.05 µm |

## Output Files

| File | Contents |
|---|---|
| `<prefix>_stack_initial.vtp` | Surface mesh before oxidation |
| `<prefix>_stack_initial.vtu` | Volume mesh before oxidation |
| `<prefix>_stack_after.vtp` | Surface mesh after oxidation |
| `<prefix>_stack_after.vtu` | Volume mesh after oxidation |
| `<prefix>_stack_after_oxide_fields.vtp` | Oxide concentration, pressure, and stress fields |

Open `.vtp` files in ParaView to visualize the surface geometry and per-point fields.

## What to Look For

- **Fin corner rounding**: oxidation rate at convex corners (fin top corners) is
  slightly enhanced because the reaction interface has higher curvature there.
  After sufficient time the corners round off noticeably.
- **Sidewall vs top rate**: for Si(100) wafers the sidewalls expose (110)-like faces
  that oxidize ~1.45× faster than the (100) top surface. This difference becomes
  visible as the oxide shell is thicker on the sides than on the top.
- **Substrate lift**: the flat substrate around the fin also grows an oxide layer;
  the Si/SiO2 boundary recedes below y = 0.
- **Volume conservation**: the `OxConcentration` field in the oxide fields VTP
  shows the oxidant gradient from the gas-phase boundary down to the Si interface.

## Solver

Each time step solves two coupled PDEs on a Cartesian grid extracted from the oxide
level-set narrow band:

**1 — Oxidant diffusion** (Deal-Grove model)

```
∇·(D(p) ∇C) = 0   inside oxide
```

with Robin BCs: transfer coefficient at the gas/oxide surface, reaction rate
`k(σ_nn) · C / N` at the Si/SiO2 interface. The diffusivity falls with the oxide
pressure and the reaction rate with the compressive normal stress at the interface
(activation volumes V_D and V_k), making growth self-limiting in tight geometries.
Solved with BiCGSTAB and a smoothed-aggregation AMG preconditioner (AMGCL), on the
CPU or the GPU.

**2 — Oxide mechanics** (incompressible Stokes flow with Maxwell viscoelasticity)

```
−∇p + ∇·s = 0         (momentum)
∇·u = 0               (incompressible oxide)
```

The volume expansion β = 2.27 enters as a velocity boundary condition at the Si/SiO2
interface and drives the oxide outward while the Si recedes. The deviatoric stress s is Maxwell viscoelastic (τ_relaxation = η/G) with a
stress-dependent (Eyring) viscosity. Velocity and pressure are solved together on a
staggered grid (sparse LU in 2-D, AMGCL in 3-D), repeated until viscosity and flow
agree.

**Coupling loop**

The oxide stress (pressure and interface normal stress) sets D and k in the diffusion
solve; the concentration sets the expansion velocity at the interface. An Aitken Δ²
outer loop iterates the two until they agree.

**Orientation correction**

The linear Deal-Grove rate constant is multiplied per-face by a crystal-normal
correction factor that captures the Si(100)/Si(110)/Si(111) rate anisotropy. For
Si(100) the exposed sidewalls are (110)-like and oxidize ~1.45× faster than the top
(100) surface.

**Time stepping**

CFL-limited explicit stepping: each accepted step satisfies
`v_max · Δt / gridDelta < 0.499`. The step grows up to 2× per accepted step and
halves on convergence failure (up to 16 retries).

**GPU acceleration**

If ViennaLS is built with `VIENNALS_USE_GPU=ON`, `useGpu=gpu` (or `auto`: the GPU
when one is available) solves the diffusion system and, in 3-D, the velocity-pressure
system on the GPU, from 20 000 unknowns: the same AMGCL solvers as on the CPU, on
AMGCL's CUDA backend. Assembly, the AMG setup and the level-set work stay on the CPU;
the 2-D mechanics (sparse LU) always runs on the CPU.

See [`docs/OxidationSolver.md`](../../ViennaLS/docs/OxidationSolver.md) in the
ViennaLS repository for the full mathematical and implementation details.
