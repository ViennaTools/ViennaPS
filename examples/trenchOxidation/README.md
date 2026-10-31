# trenchOxidation

Simulates thermal oxidation inside a rectangular Si trench. Oxide grows on the
trench floor, both sidewalls, and the flat substrate surface surrounding the
trench opening.

## Geometry

```
 ──────┬────────────┬──────  y = 0  (substrate top)
       │   trench   │
       │← trench W →│
       │            │
       └────────────┘        y = −trenchDepth  (trench floor)
         Si substrate
```

The REFLECTIVE boundary at x = 0 mirrors the domain, so the simulation represents
a half-trench; output meshes show the full symmetric structure. In 3D the trench
is extruded uniformly along Z (slot geometry).

## Configuration Parameters

All lengths are in **micrometers (µm)**, time in **hours (hr)**, pressure in **atm**.

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | `2` | Simulation dimensionality: `2` or `3` |
| `numThreads` | `16` | OpenMP thread count |
| `gridDelta` | `0.005` | Cartesian grid spacing (µm) |
| `xExtent` | `0.6` | Half-width of the domain in X (µm) |
| `zExtent` | *(= xExtent)* | 3D only: half-depth in Z (µm) |
| `trenchWidth` | `0.3` | Width of the trench opening (µm) |
| `trenchDepth` | `0.5` | Depth of the trench below y = 0 (µm) |
| `oxideThickness` | `0.0` | Oxide on the silicon at t = 0 (µm); 0 = bare silicon. Thinner than three grid cells, it first grows by the 1-D Deal-Grove law (time charged) until the solver resolves it |
| `oxidationTime` | `0.2` | Total oxidation time (hr) |
| `timeStep` | `0.025` | Maximum internal time step (hr); actual steps are CFL-limited |
| `temperature` | `1000.` | Furnace temperature (°C); valid range 700–1200 °C |
| `pressure` | `1.` | Ambient pressure (atm); scales both B and B/A linearly |
| `oxidant` | `wet` | `wet` (H₂O) or `dry` (O₂) |
| `orientation` | `100` | Crystal orientation: `100`, `110`, `111`, or `poly` |
| `maxGridPoints` | *(5M)* | Optional cap on Cartesian solve grid nodes |
| `useGpu` | `cpu` | Large linear solves on the GPU: `cpu`, `gpu` or `auto` (GPU requires `VIENNALS_USE_GPU=ON`) |
| `outputPrefix` | `ps_trench_oxidation` | Prefix for all output file names |

### Grid Sizing Guide

The diffusion solve scales as O((L/δ)^D): halving `gridDelta` increases solve
time ~4× in 2D and ~8× in 3D.

| Scenario | `gridDelta` |
|---|---|
| 2D quick test | 0.05 µm |
| 2D production | 0.005 µm |
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

- **Trench pinch-off**: at long oxidation times the oxide growing from both sidewalls
  eventually meets in the centre, closing the trench. The aspect ratio
  (`trenchDepth / trenchWidth`) controls how quickly this happens.
- **Stress-limited growth at the floor**: the confined trench floor accumulates
  compressive stress that suppresses the reaction rate via the activation-volume
  coupling. The `OxPressure` field in the oxide fields VTP visualises this directly;
  compare it against the flat substrate outside the trench.
- **Sidewall vs floor rate**: for Si(100) wafers the sidewalls are (110)-like and
  oxidize ~1.45× faster than the floor ((100)-like). The oxide shell inside the
  trench will be noticeably thicker on the sides than the floor at early times.
- **Concentration gradient**: the `OxConcentration` field drops steeply from the
  trench opening (C ≈ C*) to the bottom of the trench, showing how oxidant
  transport limits growth at the floor.
- **Corner rounding**: the sharp trench corners at the top (sidewall meets substrate)
  and at the bottom (sidewall meets floor) round as oxidation proceeds.

## Tips

- For narrow, deep trenches (`depth/width > 3`) reduce `timeStep` or increase
  `pressure` to avoid premature pinch-off artifacts. The CFL limiter handles most
  cases automatically, but very thin oxide shells in a pinched trench can cause the
  diffusion node count to drop to zero, which terminates the step early.
- The `maxGridPoints` cap is helpful for 3D runs: a 0.1 µm grid with
  xExtent = zExtent = 0.6 µm and trenchDepth = 0.5 µm gives ~400k nodes, which
  completes in a few minutes on a modern workstation.

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
(activation volumes V_D and V_k). In the trench geometry this effect is most
pronounced at the floor and sidewall junctions, where the oxide is mechanically
confined. Solved with BiCGSTAB and a smoothed-aggregation AMG preconditioner (AMGCL), on the
CPU or the GPU.

**2 — Oxide mechanics** (incompressible Stokes flow with Maxwell viscoelasticity)

```
−∇p + ∇·s = 0         (momentum)
∇·u = 0               (incompressible oxide)
```

The volume expansion β = 2.27 enters as a velocity boundary condition at the Si/SiO2
interface and drives the oxide toward the trench interior. The deviatoric stress s is Maxwell viscoelastic (τ_relaxation = η/G) with a
stress-dependent (Eyring) viscosity. Velocity and pressure are solved together on a
staggered grid (sparse LU in 2-D, AMGCL in 3-D), repeated until viscosity and flow
agree.
High compressive pressures in the trench corners are visible in the
`OxMechanicsPressure` field of the `*_oxide_fields.vtp` output.

**Coupling loop**

The oxide stress (pressure and interface normal stress) sets D and k in the diffusion
solve; the concentration sets the expansion velocity at the interface. An Aitken Δ²
outer loop iterates the two until they agree.

**Orientation correction**

The linear Deal-Grove rate constant is multiplied per-face by a crystal-normal
correction. For Si(100) wafers the trench sidewalls are (110)-like and oxidize
~1.45× faster than the (100) floor.

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
