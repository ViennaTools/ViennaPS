# LOCOS Oxidation

Simulates Local Oxidation of Silicon (LOCOS), the classical process for
field-oxide isolation in CMOS technology. A silicon nitride (Si₃N₄) pad mask
blocks oxidation on the protected side; the open window oxidizes freely. At
the mask edge, lateral diffusion of oxidant under the nitride produces the
characteristic **bird's beak**: a wedge-shaped oxide intrusion that tapers from
the full field-oxide thickness in the open window to nothing under the center
of the mask.

The ViennaPS `Oxidation` model auto-detects the `Si3N4` material and activates
the full LOCOS physics. At the ViennaLS level the `Oxidation<T,D>` wrapper
class orchestrates the per-timestep coupled solves. This document covers both
the physical model and the complete numerical implementation.

---

## Why LOCOS Is Fundamentally 2D/3D

### The Failure of a 1D Model

The Deal-Grove model predicts oxide thickness on a flat, unconstrained surface.
In LOCOS, three effects simultaneously break the 1D assumption:

1. **Lateral oxidant diffusion.** Oxidant molecules diffuse laterally under the
   mask edge, reaching the Si/SiO₂ interface in a region that is nominally
   blocked. The supply is geometry-dependent: it depends on the mask thickness,
   the pad oxide thickness, and the distance from the mask edge. No 1D formula
   can capture this spatial variation.

2. **Mechanical constraint on volume expansion.** When the oxide grows under the
   mask edge it cannot expand freely upward — the nitride is there. The expansion
   is forced laterally, building compressive stress. This stress reduces the
   effective reaction rate (Sutardja–Oldham, 1989) and deforms the mask itself.
   **Higher compressive stress → lower oxidation rate** is the primary
   self-limiting mechanism that shapes the bird's beak profile.

3. **Mask bending.** The growing oxide pushes the nitride upward from below.
   The nitride deflects at the mask edge, producing a curved contact profile
   that changes the oxide/mask contact geometry and traction distribution as
   oxidation proceeds. Neglecting mask bending underestimates the bird's beak
   penetration by 20–30%.

---

## Three-Level-Set Representation

ViennaPS represents the LOCOS structure with three level sets:

| Level set | Material interface | Sign convention |
|---|---|---|
| φ\_Si  | Si/SiO₂ reaction interface  | φ\_Si > 0 inside oxide |
| φ\_amb | SiO₂/ambient free surface   | φ\_amb < 0 inside oxide |
| φ\_mask | Si₃N₄ mask                 | φ\_mask < 0 inside nitride |

The oxide region at any time is:
```
{ x : φ_Si(x) ≥ 0  AND  φ_amb(x) ≤ 0  AND  φ_mask(x) > 0 }
```

During one time step all three interfaces move. Their motion is governed by the
coupled physics solves described below.

---

## LOCOS Physics — Coupled Solves Per Time Step

### Solve 1 — Oxidant Diffusion (Steady-State Deal-Grove PDE)

Inside the oxide band (nodes satisfying the region condition above, excluding
nodes inside the nitride), the steady-state diffusion equation is solved on a
fixed Cartesian grid:

```
∇ · (D_eff ∇C) = 0
```

where `D_eff = D · exp(−(p − p_ref) · V_D / (k_B · T))` (see stress coupling
below; with `V_D = 0`, the diffusivity D is constant).

**Boundary conditions:**

- **At φ_Si (reaction BC):**
  `-D ∂C/∂n = k_eff(p) · C`
  Oxidant is consumed by a first-order surface reaction. The effective rate
  `k_eff(p) = k · exp(−(p − p_ref) · V_k / (k_B · T))` decreases with
  compressive pressure.

- **At φ_amb (gas-transfer BC):**
  `-D ∂C/∂n = h · (C* − C)`
  Oxidant enters from the ambient gas. With `h → ∞` this reduces to the
  Dirichlet condition `C = C*` (equilibrium concentration) at the free surface.

- **At φ_mask (mask BC):**
  Zero-flux Neumann: `D ∇C · n̂ = 0`.
  The nitride is a perfect oxidant block; no oxidant enters from the mask side.
  With `maskTransferCoefficient > 0` this generalises to a Robin condition.

**Stencil assembly with embedded boundary distances:**

When a Cartesian edge from a node at grid index `i` to its neighbor exits the
oxide, the level-set zero crossing gives a sub-grid boundary point at distance

```
d = h · |φ_inside| / (|φ_inside| + |φ_outside|)
```

clamped below by `minBoundaryDistance · h` to avoid singular stencils.
For a node with sub-grid distances `d₋` and `d₊` along one axis the
second-derivative coefficients are:

```
α₊ = 2D / (d₊ · (d₋ + d₊))
α₋ = 2D / (d₋ · (d₋ + d₊))
```

Each boundary returns a `(nodeCoefficient, constant)` pair encoding
`C_boundary = nodeCoefficient · C₀ + constant`:

| Boundary type | nodeCoefficient | constant |
|---|---|---|
| Interior neighbor | 0 | C\_neighbor |
| Reaction (dist d, g = D/d) | g / (g + \|n\_a\| · k\_eff) | 0 |
| Ambient (dist d, g = D/d) | g / (g + h)      | h · C\* / (g + h) |
| Mask (h\_m = 0, zero-flux) | 1               | 0 |
| Out-of-bounds | 1 | 0 |

On a reaction arm only the axis component of the normal flux is available, so
`k_eff` is scaled by `|n_a|`, the axis component of the Si normal (exact on
tilted fronts). The stencil contributions update `rhs += α · constant` and
`diag += α · (1 − nodeCoefficient)`. The resulting sparse system is solved by
BiCGSTAB with a smoothed-aggregation AMG preconditioner (AMGCL), on the CPU or,
with `useGpu=gpu|auto`, on the GPU for systems of at least 20 000 nodes.

**Crystal-orientation reaction rate:**

The selected wafer orientation sets the baseline Deal-Grove `B/A` value. The
local surface normal then applies a continuous per-face correction:

```
k(n̂) = k_base · [1 + (r − 1) · (1 − (n̂ · ê_axis)²)]
```

where `ê_axis` is the wafer normal (`crystalAxis`) and `r` is the ratio between
the perpendicular-face `B/A` and the baseline wafer `B/A`. The built-in rates use
the ladder `(100):(110):(111) = 1 : 1.45 : 1.68`, so the per-face ratios are:

| Wafer orientation | Baseline `B/A` | Perpendicular-face ratio `r` |
|---|---|---|
| `Si100` | (100) | 1.45 |
| `Si110` | (110) | 1 / 1.45 |
| `Si111` | (111) | 1 / 1.68 |
| `PolySi` | (100), isotropic | 1 |

The local Si normal `n̂` is computed from the gradient of φ\_Si at each boundary
node.

**Reaction and expansion velocities:**

The Si/SiO₂ interface recedes at speed:
```
v_Si = velocitySign · k_eff · C / (N · γ)
```
(`velocitySign = −1` → interface moves into silicon). The local volume
expansion velocity fed to the deformation solver at each Si/SiO₂ crossing is:
```
v_exp = ((γ − 1) / γ) · k_eff · C / N
```
directed along the outward reaction-interface normal. For γ = 2.27:
- Si fraction: 1/γ ≈ 0.441
- Ambient fraction: (γ−1)/γ ≈ 0.559

**Warm-start (concentration cache):**

After each `apply()` call the concentration field is serialized into a hash map
from grid index to scalar value. On the next `apply()` call, nodes are
initialized from this cache instead of the default equilibrium concentration, so
the iterative solver starts from the previous profile, which changes little
between substeps.

---

### Solve 2 — Oxide and Nitride Deformation (Quasi-Static Stokes Flow)

The growing oxide is treated as an incompressible Maxwell viscoelastic fluid;
the nitride mask joins the same solve as a compressible Maxwell body (see
below). One
quasi-static Stokes problem is solved for both materials:

```
oxide:    ∇·(2 η_impl dev(D)) − ∇p = −∇·(d s_start),   ∇·v = 0
nitride:  ∇·σ = 0,   σ = σ_start + Δt (μ (∇v + ∇vᵀ) + λ (∇·v) I)
```

with the Maxwell quantities defined below.

**Staggered (MAC) grid.** The pressure sits at the oxide grid points of the
level-set grid, each velocity component on the midpoint of the grid edge along
its direction, and the shear stress on the cell corners (edges in 3-D). A
pressure difference between neighbouring points drives the velocity between
them directly, so there is no checkerboard pressure mode (a collocated grid has
one, which a thick, stiff nitride drives to tens of GPa).

**Boundary conditions** (the interfaces come from the signed-distance
functions and do not follow the grid):

- **Reaction interface:** a velocity in Si takes a ghost value, linear through
  the crossing where the oxide moves with the expansion velocity `v_exp` along
  the interface normal.
- **Free surface:** zero traction. The stress is extrapolated to its
  traction-free (purely tangential) value at the sub-grid crossing, with the
  surface normal from the level set.
- **Symmetry planes** (reflective domain boundaries): no normal velocity, no
  shear.

**Solve.** The velocity–pressure system is solved coupled and exactly: sparse
LU (Eigen) in 2-D, AMGCL (FGMRES with a Schur-complement pressure-correction
preconditioner) in 3-D. With the stress-dependent viscosity the solve is
repeated with updated viscosities (Anderson-accelerated) until the largest
velocity change, relative to the largest velocity, is below
`mechanicsTolerance` (default 10⁻⁴), at most `mechanicsIterations` times.

**Maxwell viscoelastic deviatoric stress:**

The symmetric strain-rate tensor is:
```
D_ij = 0.5 · (∂v_i/∂x_j + ∂v_j/∂x_i)
```

Over one substep Δt the deviatoric stress follows the Maxwell law
`ds/dt + s/τ = 2G·dev(D)`, integrated exactly for constant D:
```
s_new = d · s_start + (1 − d) · 2η · dev(D),     d = exp(−Δt/τ)
```

where `Δt = stressTimeStep` and `τ = viscosity / shearModulus` (or
`stressRelaxationTime` if specified directly). With neither set the oxide is
purely viscous (`s = 2η·dev(D)`). The full Cauchy stress is `σ = −p I + s`.

`s_start` is the stored stress at the start of the substep and stays fixed for
the whole mechanics solve. The momentum solve treats the instantaneous part
implicitly, with modulus `η_impl = (1 − d)·η` (about `G·Δt` when `Δt ≪ τ`), and
the decayed memory `d·s_start` explicitly. At the end of the substep `s_new` is
carried with the oxide velocity (semi-Lagrangian) and stored on the
`ambientInterface` level-set `pointData` by `writeFieldsToLevelSet()` (the
shear components on the corners of the staggered grid, as `OxCornerShear` and,
in the nitride, `NitCornerShear` on the mask); the next substep starts from it.

**Stress-dependent viscosity** (`setShearActivationVolume`, config `shearActivationVolume` V_c > 0; the presets use
300 Å³, the common TCAD default):

```
η(τ) = η₀ · x / sinh(x),     x = τ · V_c / (2 k T),     τ = √J₂ of the deviatoric stress
```

(Sutardja & Oldham, IEEE Trans. Electron Devices 36, 2415 (1989)). Below the
critical shear stress 2kT/V_c (0.117 GPa at 1000 °C) the viscosity stays near
η₀; above it, it falls exponentially, and the relaxation time η/G with it. A
linear viscous incompressible oxide has a pressure singularity at sharp concave
corners that grows as the grid is refined; the stress-dependent viscosity
bounds it. The viscosity lives where the stress is evaluated: at the grid
points (normal stress) and on the corners of the staggered grid (shear stress).
Each solves η = η₀ f(|s_end(η)|) for its current strain rate (bisection in
log η; unique root), and the viscosity iteration repeats the velocity–pressure
solve until viscosities and flow agree. The momentum equation carries the full
∇·(2ηε̇') with the local viscosity. The viscosity factor η/η₀ is written as
`OxViscosity`, the pressure (Pa, compression positive) as
`OxMechanicsPressure`.

**Nitride mask (monolithic oxide + nitride mechanics):**

There is no separate mask solve and no oxide/mask coupling loop: the grid
points inside the nitride join the deformation solve (equations above).

- Oxide and nitride share one velocity field, so velocity and traction are
  continuous across their interface by construction: the expanding oxide
  pushes and drags the nitride, and the nitride's stiffness pushes back.
- The nitride has no pressure unknown (compressible, ν = 0.27).
- The nitride creeps (Senez et al., IEEE TED 43, 720 (1996)): its mean
  stress is elastic, its deviatoric stress relaxes over τ = η_N/μ (about
  1.3 h at 1000 °C) with an Eyring viscosity η_N·x/sinh x (V_N = 170 Å³ at
  1000 °C), like the oxide's.
- `σ_start` is the nitride stress at the start of the substep (memory).
  It is stored on the mask level set (`NitStressR0..2`), carried with the
  moving nitride, and written for visualisation as `MaskStressR0..2` and
  `MaskVelocity` in `*_mask_fields.vtp`.
- Each surface is advected with its own material's velocity (oxide top from
  oxide nodes, mask from nitride nodes).
- The nitride is supported by the oxide it is bonded to, welded to bare Si
  and held by the symmetry plane; there is no other clamp.
- The mechanics of this 2-D example runs on the CPU (sparse LU); `useGpu=gpu`
  (or `auto`) moves the diffusion solve to the GPU from 20 000 nodes, and in
  3-D also the velocity-pressure solve.

On this example (0.1 h) the oxide grown per silicon consumed is 2.32 in the
bird's beak (−0.3 < x < 0 µm), 2.28 at the edge (0 < x < 0.1 µm) and 2.27 in
the open window, 2.275 overall (target 2.27): oxide flows from the edge into
the beak. The oxide pressure stays within about ±2 GPa; the nitride carries
−5.3 to +1.9 GPa in bending.

With `maskBarrierOnly=true` the mask is withheld from the mechanics: the oxide
expands as at a free surface and carries the mask along (exact for a
full-width mask; the mode for flat barrier-coating validation).

**Parallelism:**

The assembly of the velocity–pressure system, the strain rates and the
viscosity update are parallelised with OpenMP (one row per face, point or
corner; all data-independent).

**Free-surface velocity — local projection:**

`getVectorVelocity` returns the full Stokes field V(x) for advecting φ_amb.
Using the vector field (not just the surface-normal component) is essential at
the bird's beak corner, where the lateral oxide displacement is comparable to
the vertical displacement.

`getScalarVelocity` on the diffusion field returns the Si-interface speed
`v_Si = velocitySign · k_eff · C / (N · γ)` evaluated by finding the nearest
Si/SiO₂ level-set crossing along the query normal — a local projection that
avoids a global average.

---

### Solve 3 — Diffusion–Deformation Coupling Loop

Since the diffusion solve depends on `k_eff(p)` and the deformation solve
produces `p`, they are iterated in a fixed-point loop (`OxidationModel`):

```
for up to couplingParams.maxIterations:
    diffusionField->apply()        // solve C using current k_eff(p)
    deformationField->apply()      // solve v, p using current C
    for each deformation node:
        p_relaxed = relax * p_new + (1−relax) * p_old
        diffusionField->setPressure(index, p_relaxed)
    residual = max|Δp| / max(max|p|, k_B T / V)
    if residual < couplingParams.tolerance: break
diffusionField->apply()            // one final solve at converged state
deformationField->apply()
```

The feedback `k_eff(p)` is the mechanism by which compressive stress at the
mask edge reduces oxidant consumption and limits bird's beak growth. Without
this coupling the bird's beak profile would be overestimated.

---

### Mandatory Boolean Clips

Before and after the level-set advections, a boolean RELATIVE_COMPLEMENT
clips the ambient interface against the mask:

```
ambientInterface = ambientInterface \ maskInterface
```

**Pre-advection clip:** Removes any overlap of the ambient level set with the
mask before the interfaces are advected, so each grid point belongs to one
material.

**Post-advection clip:** The ambient interface can drift slightly into the mask
volume during the advection step, especially near the bird's beak where the
mask edge moves. The post-advection clip corrects any penetration before the
geometry is used in the next step.

Both clips always execute inside each time step — they are structural, not
conditional on interface proximity.

---

## Acceleration Strategies

### CFL-Limited Adaptive Time Stepping

Each call to `applyCFLLimited(requestedTime, cflFactor)` performs:

1. **Solve at `requestedTime`** (predictor solve): run the coupled solves
   with `stressTimeStep = requestedTime`.
2. **Compute max velocity** across the diffusion and deformation velocity
   fields (the deformation field includes the nitride).
3. **Compute CFL step:**
   `actual_dt = min(requestedTime, cflFactor · Δx / max_velocity)`
4. **If CFL reduced the step**: re-run the solves with
   `stressTimeStep = actual_dt` (corrector solve) to get the correct
   viscoelastic stress at the actual step size.
5. **Advect** the three interfaces by `actual_dt`.

The initial seed step for the outer CFL loop is pre-estimated from the Deal-Grove
B/A rate:
```
seed_dt = cflFactor · Δx / ((β−1)/β · B/A)
```
Subsequent steps reuse the last solved max velocity so the CFL condition adapts
as the oxide thickens (velocity slows down over time).

### Warm-Start Persistence Across Substeps

Three fields are persisted to level-set `pointData` before each advection and
remapped by `lsAdvect + lsInterior`:

| Field | Level set | `pointData` key |
|---|---|---|
| Oxidant concentration C(x) | `ambientInterface` | `"OxConcentration"` |
| Oxide pressure p(x) | `ambientInterface` | `"OxPressure"` |
| Oxide velocity V(x) + stress s(x) | `ambientInterface` | `"OxVelocity"`, `"OxStressR0..2"` |
| Nitride stress σ\_start(x) | `maskInterface` | `"NitStressR0..2"` |

On the following substep the concentration is the diffusion solver's initial
guess and the stresses are the mechanics' step-start (memory) stresses.

### OpenMP Parallelism

The linear solves run in AMGCL (OpenMP on the CPU, CUDA on the GPU). The
mechanics assembly, strain rates and viscosity updates are OpenMP loops (one
row per face, point or corner).

---

## Per-Time-Step Workflow

```
1.  Create OxidationDiffusion with mask exclusion, concentration warm-start
2.  Create OxidationDeformation with the nitride as part of the solve and
    the stress history of the last step
3.  OxidationModel::apply()            (diffusion + deformation coupling loop)
4.  CFL check; re-solve at the CFL-limited step if needed
5.  diffusionField->markSolved()        (prevents parallel re-entry from lsAdvect)
6.  BooleanOperation(ambient \ mask)    pre-clip
7.  Persist fields to pointData         (warm-start for next substep)
8.  Advect φ_amb with the oxide nodes' velocity
9.  Advect φ_Si with diffusion velocity
10. Advect φ_mask with the nitride nodes' velocity
11. lsInterior(φ_mask), BooleanOperation(ambient \ mask)   post-clip
12. lsInterior(φ_amb)                   (fill interior for warm-start)
```

---

## The ViennaPS `Oxidation` Model

When the ViennaPS domain contains a `Material::Si3N4` layer, `ps::Oxidation`
automatically activates LOCOS physics. No additional class or flag is needed.

### C++ Usage

```cpp
// Domain with Si, SiO2 (pad oxide), and Si3N4 (mask)
auto domain = ps::Domain<double, 2>::New();
domain->insertNextLevelSetAsMaterial(siLS,   ps::Material::Si,    false);
domain->insertNextLevelSetAsMaterial(oxLS,   ps::Material::SiO2,  false);
domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);

auto model = ps::SmartPointer<ps::Oxidation<double, 2>>::New();
model->setTemperature(1000.);          // °C
model->setOxidant(ps::OxidantType::Wet);
model->setPressure(1.0);               // atm
model->setOrientation(ps::SiliconOrientation::Si100);
model->setTimeStep(0.1);               // hr; max internal step and output cadence
model->setMaskParameters(
    viennals::OxidationPresets::siliconNitrideMask1000C());

// Run one time step at a time to save intermediate shapes
double elapsed = 0., total = 1.0;
while (elapsed < total) {
    double dt = std::min(0.1, total - elapsed);
    model->setTime(dt);
    model->setTimeStep(dt);
    ps::Process<double, 2>(domain, model, 0.0).apply();
    elapsed += dt;
    model->saveSurfaceMesh(domain, "locos_" + std::to_string(elapsed) + ".vtp");
}
```

### Python Usage

```python
import viennaps2d as vps
import viennals.d2 as ls

model = vps.Oxidation()
model.setTemperature(1000.)
model.setOxidant(vps.OxidantType.Wet)
model.setPressure(1.0)
model.setOrientation(vps.SiliconOrientation.Si100)
model.setTimeStep(0.1)
model.setMaskParameters(ls.OxidationPresets.siliconNitrideMask1000C())

elapsed = 0.0
while elapsed < 1.0:
    dt = min(0.1, 1.0 - elapsed)
    model.setTime(dt)
    model.setTimeStep(dt)
    vps.Process(domain, model, 0.0).apply()
    elapsed += dt
    domain.saveSurfaceMesh(f"locos_{elapsed:.2f}.vtp")
```

---

## Geometry Setup

```
Si substrate:          flat plane at y = 0
Pad SiO₂:             geometric offset of Si by padOxideThickness (0.03 µm)
Si₃N₄ mask:           box covering x ∈ [−xExtent, maskEdge=0],
                       y from (pad top − ε) to (pad top + maskThickness=0.05 µm)
Open oxidation window: x > 0 (right half of domain)
```

The small contact epsilon (10⁻⁶ µm) between the mask bottom and the pad oxide
top ensures that Cartesian stencil edges unambiguously cross the mask boundary.
Without it, nodes exactly at the mask bottom may be misclassified.

```
Domain:   x ∈ [−1, 1] µm   REFLECTIVE boundaries
          y ∈ [−1, 2] µm   INFINITE boundaries
gridDelta = 0.01 µm
```

---

## Parameters

### Process parameters (`setXxx` on `ps::Oxidation`)

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | — | °C; sets k, D via Arrhenius fits to Deal-Grove data |
| `oxidant` | — | `Wet` (H₂O) or `Dry` (O₂); changes pre-exponentials |
| `pressure` | 1.0 atm | Scales C\* proportionally (B and B/A both linear in pressure) |
| `orientation` | `Si100` | `Si100`, `Si110`, `Si111`, or `PolySi`; selects the baseline `B/A` and crystal-axis correction |
| `timeStep` | — | hr; caps each internal oxidation substep and saved shape cadence |
| `setCFLFactor` | 0.499 | Courant number; `actual_dt ≤ cflFactor · Δx / max_vel` |
| `maxGridPoints` | 5×10⁶ | Memory limit for the Cartesian oxide solve grid |
| `couplingIterations` | 8 | Max iterations for diffusion–deformation coupling loop |
| `couplingTolerance` | 10⁻⁶ | Relative pressure convergence threshold |

For dry oxidation, the built-in Deal-Grove `B/A` table uses two Arrhenius
regimes split at 950 °C: `E_a = 2.30 eV` below 950 °C and `E_a = 2.00 eV` at or
above 950 °C. Wet oxidation uses a single `B/A` activation energy. `B` is treated
as orientation-independent; `B/A` carries the `(100):(110):(111)` orientation
ratios.

### Oxidation parameters (inside ViennaLS, set by `psOxidation`)

| Parameter | Value (wet 1000 °C ⟨100⟩) | Meaning |
|---|---|---|
| `diffusionCoefficient` D | 0.157 µm²/hr | B/2 = 0.314/2 (parabolic rate constant) |
| `reactionRate` k | 0.74 µm/hr | B/A (linear rate constant) |
| `transferCoefficient` h | 100 µm/hr | Large → C ≈ C\* at ambient surface |
| `equilibriumConcentration` C\* | 1 | Normalized; scales with pressure |
| `expansionCoefficient` γ | 2.27 | SiO₂/Si molar volume ratio |
| `velocitySign` | −1 | Interface moves into silicon |
| `stressDependentOxidation` | true | Stress-dependent kinetics on/off (`setStressDependentOxidation`) |
| `reactionActivationVolume` V_k | 2.5×10⁻²⁹ m³ (25 Å³) | Reaction rate vs interface normal stress, `k = k₀ exp(σ_nn V_k / k_BT)`, retarding only; Kao-calibrated |
| `diffusionActivationVolume` V_D | 4×10⁻²⁹ m³ (40 Å³) | Diffusivity vs pressure, `D = D₀ exp(−p V_D / k_BT)`; Kao-calibrated |
| `maskTransferCoefficient` | 0 | Nitride is a perfect oxidant block |

### Deformation parameters (inside ViennaLS)

| Parameter | Value | Meaning |
|---|---|---|
| `viscosity` η | Arrhenius per oxidant; wet 3.0×10⁹ Pa·hr, dry 6.5×10¹¹ Pa·hr at 1000 °C | Low-stress oxide viscosity, `E_a` = 3.46 eV (wet) / 4.56 eV (dry); calibrated on Kao et al. 1987 cylinders with V_k = 25, V_D = 40, V_c = 300 Å³ |
| `shearModulus` G | 3×10¹⁰ Pa | Maxwell deviatoric relaxation |
| `stressTimeStep` Δt | per substep | Maxwell relaxation time step |
| `shearActivationVolume` V_c | 3×10⁻²⁸ m³ (300 Å³) | Stress-dependent viscosity η₀·x/sinh x; 0 = linear |
| `mechanicsIterations` | 300 (config) | Maximum viscosity iterations (one velocity–pressure solve each) |
| `mechanicsTolerance` | 10⁻⁴ | Relative velocity change at which the viscosity iteration stops |

### Mask parameters (`setMaskParameters` on `ps::Oxidation`)

| Parameter | Value | Meaning |
|---|---|---|
| `youngModulus` E | 270 GPa | Si₃N₄ at 1000 °C |
| `poissonRatio` ν | 0.27 | Si₃N₄ Poisson's ratio; sets λ/μ of the nitride |
| `viscosity` η_N | 1.4×10¹¹ Pa·hr at 1000 °C | Maxwell creep of the deviatoric stress (Senez 1996, Table II: 800-1100 °C); 0 = elastic |
| `shearActivationVolume` V_N | 1.7×10⁻²⁸ m³ (170 Å³) at 1000 °C | Eyring η_N·x/sinh x; 0 = linear |

Without `setMaskParameters` the mask takes the Si₃N₄ values at the process
temperature (`OxidationPresets::siliconNitrideMask`).

---

## What to Expect from the Simulation

After 0.1 hr of wet oxidation at 1000 °C, 1 atm, with 0.03 µm pad oxide:

- **Open window:** ~0.07 µm additional oxide grows; the ambient surface rises
  ~0.04 µm and the Si interface sinks ~0.03 µm.
- **Bird's beak:** Oxide extends under the mask edge, tapering from the
  open-window thickness to zero over ~2–4× the pad oxide thickness.
- **Mask deflection:** Nitride bottom curves upward by ~0.005–0.01 µm at the
  mask edge.
- **Oxidant suppression:** Concentration under the mask center is less than
  0.01% of the open-window value.

The planar Deal-Grove estimate (`estimatePlanarOxideThickness()`) gives the
open-window thickness to within ~5%, serving as a consistency check.

---

## Time-Stepping Considerations

LOCOS requires multiple physical updates because:
1. The geometry changes substantially over the oxidation time, so the coupled
   diffusion/mechanics fields must be refreshed as interfaces move.
2. Each internal step re-solves the oxide and nitride mechanics with the current
   geometry.

The `timeStep` parameter caps the maximum internal substep and sets the output
cadence. If larger than the CFL limit, the model automatically subcycles. The
config sets `timeStep = 0.1 hr` as a practical output cadence while the CFL
condition typically allows larger actual steps for thin oxides.

---

## Running the Example

### C++
```bash
cmake --build build --target locosOxidation
cd build/examples/locosOxidation && ./locosOxidation config.txt
```

### Python
```bash
python3 examples/locosOxidation/locosOxidation.py config.txt
```

Configuration keys in `config.txt` (lengths in µm, time in hours):
```
numThreads=16
gridDelta=0.01
xExtent=1.0
yMin=-1.0       yMax=2.0
padOxideThickness=0.03
maskThickness=0.05
maskEdge=0.0
oxidationTime=0.1
timeStep=0.1
temperature=1000.
pressure=1.0
oxidant=wet
orientation=100
mechanicsIterations=300
mechanicsTolerance=1e-4
couplingIterations=100
couplingTolerance=2e-2
maskYoungModulus=270e9
maskPoissonRatio=0.27
maskViscosity=-1
maskShearActivationVolume=-1
```

---

## Output Files

| File | Contents |
|---|---|
| `ps_locos_step_NNNN.vtp` | Surface mesh at each time step (Si, SiO₂, Si₃N₄) |
| `ps_locos_after.vtp` | Final surface mesh |
| `ps_locos_initial.vtp` | Initial surface mesh |

Open in ParaView. Zoom into the mask edge region at x ≈ 0 to visualize the
bird's beak profile and the nitride deflection.

---

## Implementation Files

| File | Role |
|---|---|
| `lsOxidation.hpp` | Unified oxidation orchestrator (`Oxidation<T,D>`); owns the per-step workflow |
| `lsOxidationDiffusion.hpp` | Deal-Grove diffusion field (cut-cell stencil, BiCGSTAB with AMG on the CPU or GPU) |
| `lsOxidationDeformation.hpp` | Maxwell Stokes deformation on a staggered grid (sparse LU in 2-D, AMGCL in 3-D); nitride mask in the same solve |
| `lsOxidationModel.hpp` | Pressure–concentration coupling loop |
| `lsOxidationMask.hpp` | Nitride parameters; barrier-only mask carry |
| `lsOxidationPresets.hpp` | Calibrated material presets (wet/dry 1000 °C) |
| `psOxidation.hpp` | ViennaPS wrapper; Deal-Grove Arrhenius lookup; material detection |

---

## Further Reading

- B.E. Deal and A.S. Grove, *General Relationship for the Thermal Oxidation
  of Silicon*, J. Appl. Phys. **36**, 3770 (1965).
- P. Sutardja and W.G. Oldham, *Modeling of Stress Effects in Silicon
  Oxidation*, IEEE Trans. Electron Devices **36**, 2415 (1989).
- K. Taniguchi, M. Tanaka, C. Hamaguchi, K. Imai, *Two-Dimensional Computer
  Analysis of LOCOS Process*, J. Electrochem. Soc. **137**, 1589 (1990).
- D.B. Kao, J.R. McVittie, W.D. Nix, K.C. Saraswat, *Two-Dimensional Thermal
  Oxidation of Silicon — I. Experiments*, IEEE Trans. Electron Devices **34**,
  1008 (1987).
- E.A. Irene, *Silicon Oxidation Studies: A Quantitative Investigation of the
  Pressure Retardation of the Thermal Oxidation of Silicon*, J. Electrochem.
  Soc. **125**, 1708 (1978).
- H.-H. Massoud and J.D. Plummer, *Thermal Oxidation of Silicon — II*,
  J. Electrochem. Soc. **132**, 2693 (1985).
