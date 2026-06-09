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

## Building

```bash
# From the repository root
cmake -B build -DVIENNAPS_BUILD_EXAMPLES=ON ...
cmake --build build --target finOxidation
```

## Running

```bash
# Default config (reads config.txt in CWD)
./finOxidation

# Explicit config file
./finOxidation my_config.txt
```

The Python version works identically:

```bash
python finOxidation.py            # reads config.txt
python finOxidation.py my_config.txt
```

## Configuration Parameters

All lengths are in **micrometers (µm)**, time in **hours (hr)**, pressure in **atm**.

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | `2` | Simulation dimensionality: `2` or `3` |
| `numThreads` | `16` | OpenMP thread count |
| `gridDelta` | `0.01` | Cartesian grid spacing (µm) |
| `xExtent` | `0.6` | Half-width of the domain in X (µm) |
| `yMin` | `-1.0` | Bottom of the domain in Y (µm) |
| `yMax` | `2.0` | Top of the domain in Y (µm) |
| `zExtent` | *(= xExtent)* | 3D only: half-depth in Z (µm) |
| `finWidth` | `0.2` | Width of the Si fin (µm) |
| `finHeight` | `0.5` | Height of the Si fin above the substrate (µm) |
| `oxideThickness` | `0.0` | Pre-existing oxide seed thickness (µm); a seed of `gridDelta` is used when zero |
| `oxidationTime` | `0.2` | Total oxidation time (hr) |
| `timeStep` | `0.025` | Maximum internal time step (hr); actual steps are CFL-limited |
| `temperature` | `1000.` | Furnace temperature (°C); valid range 700–1200 °C |
| `pressure` | `1.` | Ambient pressure (atm); scales both B and B/A linearly |
| `oxidant` | `wet` | `wet` (H₂O) or `dry` (O₂) |
| `orientation` | `100` | Crystal orientation: `100`, `110`, `111`, or `poly` |
| `maxGridPoints` | *(5M)* | Optional cap on Cartesian solve grid nodes |
| `outputPrefix` | `ps_fin_oxidation` | Prefix for all output file names |

### Grid Sizing Guide

| Scenario | `gridDelta` | Approx. node count |
|---|---|---|
| 2D fine | 0.01 µm | ~100k |
| 2D fast | 0.025 µm | ~16k |
| 3D practical | 0.05 µm | ~3M |
| 3D fast test | 0.1 µm | ~400k |

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

## Physics Notes

See [`docs/OxidationSolver.md`](../../ViennaLS/docs/OxidationSolver.md) in the
ViennaLS repository for the full description of the diffusion PDE, Stokes deformation
solver, pressure–concentration coupling, and CFL-limited time stepping.
