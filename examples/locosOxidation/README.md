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
four coupled physics solves described below.

---

## LOCOS Physics — Four Coupled Solves Per Time Step

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
| Reaction (dist d, g = D/d) | g / (g + k\_eff) | 0 |
| Ambient (dist d, g = D/d) | g / (g + h)      | h · C\* / (g + h) |
| Mask (h\_m = 0, zero-flux) | 1               | 0 |
| Out-of-bounds | 1 | 0 |

The stencil contributions update `rhs += α · constant` and
`diag += α · (1 − nodeCoefficient)`. The Jacobi update is:
`C_new = rhs / diag`, then relaxed: `C = relax · C_new + (1−relax) · C_old`.

Convergence: `max |C_new − C_old| < tolerance` (default 10⁻⁷).

**Crystal-orientation reaction rate:**

On faces other than (100) the reaction rate is modulated by

```
k(n̂) = k_eff · [1 + (r₁₁₁ − 1) · (1 − (n̂ · ê₁₀₀)²)]
```

where `r₁₁₁ = reactionRateRatio111` (= 1.0 → isotropic) and `ê₁₀₀` is the
wafer crystallographic axis (`crystalAxis`). The local Si normal `n̂` is
computed from the gradient of φ\_Si at each boundary node.

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
initialized from this cache instead of the default equilibrium concentration. This
reduces the Jacobi iteration count for the diffusion solve by 30–50% after the
first step, since the concentration profile changes little between substeps.

---

### Solve 2 — Oxide Deformation (Quasi-Static Stokes Flow)

The growing oxide is treated as a viscous material. The solve has three stages:

**Stage 1 — Harmonic Extension (predictor velocity)**

At each Si/SiO₂ crossing the expansion velocity `v_exp` is set as a Dirichlet
condition directed along the local reaction-interface normal. This is harmonically
extended through the oxide band by iteratively averaging each interior node over
its Cartesian neighbors until convergence (`harmonicIterations`, `tolerance`).

At mask/oxide contact crossings the current mask velocity from Solve 4 is used
as the Dirichlet value; without an attached mask field the fallback is
stationary (no-slip) at the mask face.

The result serves as the predictor velocity for Stages 2–3.

**Stage 2 — Pressure solve**

From the current velocity field the divergence `div(v)` is computed at each node
with central-difference stencils using sub-grid boundary distances. The pressure
Poisson equation

```
∇²p = −K · div(v)
```

is solved with `K = bulkModulus`. Boundary conditions:

- **Free surface (φ_amb):** Dirichlet from the traction-free condition:
  `p_surface ≈ p_ambient + n̂ · s_dev · n̂` (where `s_dev` is the
  deviatoric stress from the previous mechanics iteration; zero on the first).
- **Reaction interface (φ_Si):** Zero-normal-gradient Neumann; the velocity is
  prescribed by oxidation kinematics so no pressure penalty spring is applied.
- **Mask contact (φ_mask):** Zero-normal-gradient Neumann; mask resistance
  enters through the coupled mask mechanics solve.

Solved by point-Jacobi iteration (`pressureIterations`, `pressureTolerance`).

**Stage 3 — Stokes velocity update**

The quasi-static Stokes momentum equation is:
```
η · ∇²v = ∇p − ∇ · s_dev
```
(`η = viscosity`). The right-hand side `∇p − ∇ · s_dev` is computed with
central-difference stencils at each node. Boundary conditions:

- **Reaction interface:** Dirichlet `v = v_exp` (from Stage 1).
- **Free surface:** Ghost velocity from the traction-free condition:
  `v_ghost = 2 · v_surface_analytical − v_node`
  (second-order one-sided estimate of the traction gradient).
- **Mask contact:** Dirichlet velocity from the current mask mechanics solve.

Solved by point-Jacobi iteration (`stokesIterations`, `stokesTolerance`).

**Mechanics outer loop:**

Stages 2 and 3 are repeated for `mechanicsIterations` outer iterations
until the relative change in both pressure and velocity falls below
`mechanicsTolerance`.

**Maxwell viscoelastic deviatoric stress:**

After each velocity update the symmetric strain-rate tensor is:
```
D_ij = 0.5 · (∂v_i/∂x_j + ∂v_j/∂x_i)
```

The deviatoric stress evolves with a Maxwell relaxation law:
```
s_new = exp(−Δt/τ) · s_old + (1 − exp(−Δt/τ)) · 2η · dev(D)
```

where `Δt = stressTimeStep` and `τ = viscosity / shearModulus` (or
`stressRelaxationTime` if specified directly). Without a shear modulus (`τ → ∞`)
the oxide is purely viscous and `s_dev = 0`. The full Cauchy stress is
`σ = −p I + s_dev`.

The stress history `s_old` is persisted to the `ambientInterface` level-set
`pointData` by `writeFieldsToLevelSet()` before advection. After advection and
`lsInterior` the field is remapped to the new geometry, providing a warm-start
for the next step's deformation solve without a separate in-memory cache.

**Parallelism:**

The harmonic extension, pressure Jacobi, and Stokes Jacobi inner loops are all
parallelised with `#pragma omp parallel for reduction(max:residual)`. The
iteration-free (per-face boundary cache) design allows all nodes to be updated
simultaneously without race conditions.

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
    residual = max|Δp| / max|p|
    if residual < couplingParams.tolerance: break
diffusionField->apply()            // one final solve at converged state
deformationField->apply()
```

The feedback `k_eff(p)` is the mechanism by which compressive stress at the
mask edge reduces oxidant consumption and limits bird's beak growth. Without
this coupling the bird's beak profile would be overestimated.

---

### Solve 4 — Mask Bending (Quasi-Static Lamé Elasticity)

The nitride mask is modeled as a viscous elastic solid with temperature-dependent
creep viscosity. The Lamé momentum equation is solved inside the mask domain:

```
μ ∇²v_mask + (λ + μ) ∇(∇ · v_mask) = 0
```

The Lamé viscosity parameters are derived from the Arrhenius mask viscosity
`η(T)` and Poisson's ratio ν:

```
η(T) = η_ref · exp(E_a/R · (1/T − 1/T_ref))

μ_v = η(T) / (2(1 + ν))
λ_v = η(T) · ν / ((1 + ν)(1 − 2ν))
```

**Solve domain:**

All Cartesian grid nodes inside the mask level set (`φ_mask < 0`) within the
user-specified mask bending bounds are included.

**Contact nodes:**

A mask node is a contact node when at least one of its Cartesian faces exits
the mask toward the oxide side AND that neighbor location is inside the oxide
band (below the ambient surface, above the Si surface). The contact side is
detected from the signed gradient of φ\_mask using central differences. HRLE
far-field sentinels are clamped before differencing.

The additional ambient-band check (using the cached ambient phi) prevents mask
nodes above the oxide surface from being falsely classified as contact nodes.

**Contact boundary condition — traction from oxide stress:**

Contact nodes participate in the interior Jacobi solve, but their out-of-mask
neighbor velocities are replaced by ghosts derived from the oxide Cauchy stress
tensor at the contact face. For a contact face with outward normal n̂ (pointing
from mask into oxide):

```
t_i = Σ_j σ_oxide_ij · n̂_j       (σ_oxide = s_dev − p·I)
```

The ghost velocity is obtained from the first-order Neumann condition:

```
v_ghost_normal     = v_node_normal     + h / (λ + 2μ) · t_n          (normal)
v_ghost_tangential = v_node_tangential + h / μ        · t_tangential  (shear)
```

where `t_n = t · n̂` and `t_tangential = t − t_n · n̂`.

The contact is unilateral by default: if `t_n ≥ 0` (tensile), the ghost falls
back to the current mask-node velocity and the oxide does not pull the mask.

**Physical interpretation:**
- Larger oxide pressure → larger traction → larger ghost offset → contact node
  accelerates in the normal direction.
- Larger η(T) → larger μ and λ → smaller ghost offset for the same traction
  → stiffer mask bends more slowly.
- A thicker mask has more nodes between the contact face and the anchor, so the
  traction-driven velocity decays before reaching the top — no explicit
  compliance scaling is needed.

**Interior node update — Jacobi relaxation of the Lamé equation:**

Each iteration applies:

1. **Laplacian average** (resolves μ ∇²v):
   `v_avg = (1/count) · Σ_neighbors v_neighbor`
   Contact ghost velocities substitute for out-of-mask faces; zero-flux Neumann
   at out-of-bounds faces.

2. **Grad-div correction** (resolves (λ + μ) ∇(∇ · v)):
   `v_update += gradDivWeight · h² / (2D) · ∇(∇ · v)`
   where `gradDivWeight = (λ + μ) / max(λ + 2μ, ε)`. This term enforces
   volumetric compatibility and distinguishes the Lamé equation from pure
   Laplace smoothing.

The relaxed update is: `v = relaxation · v_update + (1 − relaxation) · v_old`.

Convergence: `max |Δv_component| < tolerance` over all nodes.

**Warm-start:**

The mask velocity field is persisted to the `maskInterface` level-set
`pointData` by `writeFieldsToLevelSet()` before advection, then remapped and
filled by `lsAdvect + lsInterior`. The `seedFromLevelSet()` call at the start
of the next `apply()` initializes node velocities from this saved field,
reducing the Jacobi iteration count significantly across time steps.

---

### The Constrained Ambient Velocity Field

`OxidationConstrainedAmbient` adapts the ambient-interface advection velocity
depending on whether a query point is under the mask or in the open window:

```
if φ_mask(x) ≤ 0  (inside nitride):
    v_amb(x) = V_mask(x)     (moves with nitride, no free oxidation)
else:
    v_amb(x) = V_oxide(x)    (moves with free oxide surface)
```

Additionally, for gap-zone points just outside the mask surface
(`−3Δx < φ_mask < 0`), a no-separation correction boosts the oxide velocity
in the interface-normal direction to match the mask, preventing a persistent
sub-grid void from opening at the bird's beak corner.

The mask phi values are cached in a flat hash map before advection begins
(`buildMaskPhiCache`), making each velocity query O(1) without per-call HRLE
iterator traversal.

---

### Mandatory Boolean Clips

Before and after the three level-set advections, a boolean RELATIVE_COMPLEMENT
clips the ambient interface against the mask:

```
ambientInterface = ambientInterface \ maskInterface
```

**Pre-advection clip:** The ambient and mask level sets must not overlap when
the constrained velocity field queries φ\_mask. Overlap makes the inside/outside
test ambiguous and corrupts the velocity field.

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

1. **Solve at `requestedTime`** (predictor solve): run all four coupled solves
   with `stressTimeStep = requestedTime`.
2. **Compute max velocity** across all three velocity fields
   (diffusion, deformation, mask bending).
3. **Compute CFL step:**
   `actual_dt = min(requestedTime, cflFactor · Δx / max_velocity)`
4. **If CFL reduced the step**: re-run all four solves with
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
| Oxide velocity V(x) + stress s_dev(x) | `ambientInterface` | `"OxVelocity"`, `"OxStress"` |
| Mask velocity V\_mask(x) | `maskInterface` | `"MaskVelocity"` |

On the following substep `seedFromLevelSet()` restores these fields as the
initial guess for each solver, eliminating the cold-start cost from the second
substep onward.

### Aitken Acceleration for the Oxide/Mask Interface Loop

The oxide/mask coupling loop is a fixed-point iteration: oxide solve → mask
solve → oxide solve → ... After the first iteration, Aitken's Δ² method
is applied to the contact-velocity residual vector to extrapolate toward the
fixed point. The relaxation factor ω is updated each iteration:

```
ω_new = −ω_old · (r_old · Δr) / (Δr · Δr)
ω     = clamp(ω_new, 0.05, 1.5)
```

where `Δr = r_new − r_old` is the residual increment. This typically achieves
convergence in 3–7 iterations instead of the 10–20 without acceleration.
The initial ω is 1 (no relaxation); subsequent values adapt based on the
curvature of the residual sequence.

### OpenMP Parallelism

All inner Jacobi loops are parallelised with OpenMP:

| Loop | Parallelism note |
|---|---|
| Diffusion Jacobi | `#pragma omp parallel for reduction(max:residual)` |
| Harmonic extension | `#pragma omp parallel for reduction(max:residual)` |
| Pressure Jacobi | `#pragma omp parallel for` (divergence) + Jacobi |
| Stokes Jacobi | `#pragma omp parallel for reduction(max:residual)` |
| Mask bending Jacobi | `#pragma omp parallel for reduction(max:residual)` |

The per-node update reads only from the previous-iteration vector and writes
to the next-iteration vector — a Jacobi (not Gauss-Seidel) update — so all
nodes are data-independent and the parallelism is race-free by construction.

---

## Per-Time-Step Workflow

```
1.  Create OxidationDiffusion with mask exclusion, concentration warm-start
2.  Create OxidationDeformation with mask contact velocity, velocity warm-start
3.  OxidationModel::apply()            (diffusion + deformation coupling loop)
4.  Create OxidationMaskBending with ambient phi cache, velocity warm-start
5.  OxidationMaskBending::apply()      (first mask bending solve)

6.  Oxide/mask interface coupling (Aitken-accelerated fixed-point):
      for iteration in 1..maskCouplingIterations:
          deformationField->setMaskVelocityField(maskBendingField)
          OxidationModel::apply()        (oxide re-solve with mask velocity BC)
          OxidationMaskBending::apply()  (mask re-solve with oxide traction BC)
          Aitken-relax the contact-velocity update
          if contact-velocity change < maskCouplingTolerance: break

7.  Create OxidationConstrainedAmbient (mask phi cache built once)
8.  diffusionField->markSolved()        (prevents parallel re-entry from lsAdvect)
9.  BooleanOperation(ambient \ mask)    pre-clip
10. Persist fields to pointData         (warm-start for next substep)
11. Advect φ_amb with constrained ambient velocity
12. Advect φ_Si with diffusion velocity
13. Advect φ_mask with mask bending velocity
14. lsInterior(φ_amb), lsInterior(φ_mask)   (fill interior for warm-start)
15. BooleanOperation(ambient \ mask)    post-clip
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
    ls::OxidationMaterials<double>::siliconNitrideMask1000C());
model->setMaskCouplingIterations(30);
model->setMaskCouplingTolerance(0.04);

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
model.setMaskParameters(ls.OxidationProcessPresets.siliconNitrideMask1000C())
model.setMaskCouplingIterations(30)
model.setMaskCouplingTolerance(0.04)

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
| `orientation` | `Si100` | Crystal anisotropy factor on k (see r₁₁₁ above) |
| `timeStep` | — | hr; caps each internal oxidation substep and saved shape cadence |
| `setCFLFactor` | 0.499 | Courant number; `actual_dt ≤ cflFactor · Δx / max_vel` |
| `maxGridPoints` | 5×10⁶ | Memory limit for the Cartesian oxide solve grid |
| `couplingIterations` | 8 | Max iterations for diffusion–deformation coupling loop |
| `couplingTolerance` | 10⁻⁶ | Relative pressure convergence threshold |

### Oxidation parameters (inside ViennaLS, set by `psOxidation`)

| Parameter | Value (wet 1000 °C ⟨100⟩) | Meaning |
|---|---|---|
| `diffusionCoefficient` D | 0.157 µm²/hr | B/2 = 0.314/2 (parabolic rate constant) |
| `reactionRate` k | 0.74 µm/hr | B/A (linear rate constant) |
| `transferCoefficient` h | 100 µm/hr | Large → C ≈ C\* at ambient surface |
| `equilibriumConcentration` C\* | 1 | Normalized; scales with pressure |
| `expansionCoefficient` γ | 2.27 | SiO₂/Si molar volume ratio |
| `velocitySign` | −1 | Interface moves into silicon |
| `reactionActivationVolume` V_k | 1.76×10⁻³⁵ m³ | Sutardja–Oldham stress correction |
| `diffusionActivationVolume` V_D | 0 | Massoud–Plummer; 0 = off |
| `maskTransferCoefficient` | 0 | Nitride is a perfect oxidant block |

### Deformation parameters (inside ViennaLS)

| Parameter | Value | Meaning |
|---|---|---|
| `viscosity` η | 10¹⁰ Pa·hr | Effective oxide viscosity at 1000 °C |
| `bulkModulus` K | 7.5×10⁸ Pa | p ← divergence coupling coefficient |
| `shearModulus` G | 3×10¹⁰ Pa | Maxwell deviatoric relaxation |
| `stressTimeStep` Δt | per substep | Maxwell relaxation time step |
| `mechanicsIterations` | 30 (config) | Pressure/velocity outer iterations |
| `pressureIterations` | 2000 (config) | Inner pressure Jacobi iterations |
| `stokesIterations` | 1000 (config) | Inner Stokes Jacobi iterations |
| `mechanicsTolerance` | 10⁻⁷ | Relative change threshold for mechanics outer loop |

### Mask parameters (`setMaskParameters` on `ps::Oxidation`)

| Parameter | Value | Meaning |
|---|---|---|
| `referenceViscosity` η\_ref | 5×10¹¹ Pa·hr | Creep viscosity at referenceTemperature |
| `referenceTemperature` | 1273.15 K | Reference temperature for Arrhenius law |
| `creepActivationEnergy` E\_a | 0 J/mol | Arrhenius exponent; 0 = temperature-independent |
| `poissonRatio` ν | 0.27 | Si₃N₄ Poisson's ratio; sets λ/μ in Lamé equation |
| `unilateralContact` | true | Oxide can push but not pull the mask |
| `relaxation` | 1.0 | Bending Jacobi under-relaxation |
| `tolerance` | 10⁻⁸ | Bending Jacobi convergence threshold |
| `maxIterations` | 10000 | Max Jacobi iterations for bending solve |

### Mask coupling parameters (outer oxide/mask loop)

| Parameter | Value (config) | Meaning |
|---|---|---|
| `maskCouplingIterations` | 30 | Max Aitken-accelerated iterations per step |
| `maskCouplingTolerance` | 0.04 | Relative contact-velocity change threshold |

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
1. The geometry changes substantially over the oxidation time, so coupled
   diffusion/mechanics/contact fields must be refreshed as interfaces move.
2. Each internal step re-solves the mask bending with the current oxide geometry.

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
mechanicsIterations=30
pressureIterations=2000
stokesIterations=1000
couplingIterations=30
couplingTolerance=1e-4
maskCouplingIterations=30
maskCouplingTolerance=0.04
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
| `lsOxidationDiffusion.hpp` | Deal-Grove diffusion field (Jacobi, embedded BCs) |
| `lsOxidationDeformation.hpp` | Stokes deformation field (harmonic + pressure + Stokes) |
| `lsOxidationModel.hpp` | Pressure–concentration coupling loop |
| `lsOxidationMask.hpp` | Mask bending solver and constrained ambient velocity |
| `lsOxidationMaterials.hpp` | Calibrated material presets (wet/dry 1000 °C) |
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
