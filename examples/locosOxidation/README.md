# LOCOS Oxidation

Simulates Local Oxidation of Silicon (LOCOS), the classical process for
field-oxide isolation in CMOS technology. A silicon nitride (Si₃N₄) pad mask
blocks oxidation on the protected side; the open window oxidizes freely.
At the mask edge, the lateral diffusion of oxidant under the nitride produces
the characteristic **bird's beak**: a wedge-shaped oxide intrusion that tapers
from the full field-oxide thickness in the open window to nothing under the
center of the mask.

The ViennaPS `Oxidation` model auto-detects the Si₃N₄ material and activates
the full LOCOS physics described below.

---

## Why LOCOS Is Fundamentally 2D/3D

### The Problem with a 1D Model

The Deal-Grove model predicts oxide thickness on a flat, unconstrained surface.
In LOCOS, three effects break the 1D assumption simultaneously:

1. **Lateral oxidant diffusion.** Oxidant molecules diffuse laterally under the
   mask edge, reaching the Si/SiO₂ interface in a region that is nominally
   blocked. The supply is geometry-dependent: it depends on the mask thickness,
   the pad oxide thickness, and the distance from the mask edge.

2. **Mechanical constraint on volume expansion.** When the oxide grows under
   the mask edge, it cannot expand freely upward — the nitride is there. The
   expansion is forced laterally, building up compressive stress. This stress
   reduces the effective reaction rate (Sutardja–Oldham, 1989) and deforms
   the mask itself. **Higher compressive stress → lower oxidation rate** is
   the primary self-limiting mechanism that defines the bird's beak profile.

3. **Mask bending.** The growing oxide pushes up on the nitride from below.
   The nitride deflects at the mask edge, producing a curved contact profile
   that changes the contact geometry and traction distribution as oxidation
   proceeds. Neglecting mask bending underestimates the bird's beak penetration
   by 20–30%.

None of these can be described by a 1D model. LOCOS is inherently a 2D (or 3D)
geometry simulation problem.

---

## Three-Level-Set Representation

ViennaPS represents the LOCOS structure with three level sets:

| Level set | Material interface | Sign convention |
|---|---|---|
| φ_Si | Si/SiO₂ reaction interface | φ_Si > 0 inside oxide |
| φ_amb | SiO₂/ambient free surface | φ_amb < 0 inside oxide |
| φ_mask | Si₃N₄ mask | φ_mask < 0 inside nitride |

The oxide region at any point in time is:
```
{x : φ_Si(x) ≥ 0  AND  φ_amb(x) ≤ 0  AND  φ_mask(x) > 0}
```
(inside the oxide, below the free surface, and outside the nitride).

During one time step all three interfaces move. Their motion is governed by
four coupled physics solves, described next.

---

## LOCOS Physics — Four Coupled Solves Per Time Step

### Solve 1 — Oxidant Diffusion

Inside the oxide band (excluding nodes inside the nitride), the steady-state
diffusion equation is solved:

```
∇ · (D_eff ∇C) = 0
```

Boundary conditions:
- **At φ_Si (reaction BC):** `-D ∂C/∂n = k_eff(p) · C`  
  Oxidant consumed at the Si surface.
- **At φ_amb (gas-transfer BC):** `-D ∂C/∂n = h · (C* − C)`  
  Oxidant enters from the ambient gas.
- **At φ_mask (mask BC):** zero-flux Neumann  
  The nitride is a perfect oxidant block (no oxidant crosses the nitride face).

The lateral gradient of C under the mask edge provides the source of
bird's beak growth. Where the mask overhangs the Si, the oxidant
must diffuse a longer path to reach the interface, so C(x) is lower there
and oxidation is slower. The spatial profile of C(x) directly shapes the
bird's beak geometry.

The effective reaction rate `k_eff(p)` and diffusivity `D_eff(p)` depend on
the local oxide pressure from the mechanics solve. Higher compressive pressure
(greatest under the mask) reduces both, making oxidation self-retarding in
the region where it is most constrained.

### Solve 2 — Oxide Deformation (Quasi-Static Stokes Flow)

Volume expansion at the Si/SiO₂ interface drives viscous flow throughout the
oxide. The Stokes momentum equation is solved:

```
η ∇²v = ∇p − ∇ · s_dev
```

where v is the oxide velocity field, η the viscosity, p the pressure, and
s_dev the Maxwell viscoelastic deviatoric stress.

Key boundary conditions:
- **At φ_Si:** Dirichlet — the expansion velocity `v_exp = ((γ−1)/γ)·k_eff·C/N`
  directed along the Si outward normal.
- **At φ_amb (open window):** Traction-free — the oxide surface in the open
  window can deform freely.
- **At φ_mask contact face:** The oxide velocity is coupled to the mask
  bending velocity from Solve 4 below. This enforces velocity continuity at
  the oxide/mask interface: the oxide cannot penetrate the nitride, and the
  oxide pushes the mask by the same velocity as the oxide moves.

The resulting vector velocity field V(x) advects the φ_amb level set. Using
the full vector field (not just the surface-normal component) is essential at
the bird's beak corner, where the lateral displacement of the oxide surface
underneath the mask edge is comparable to the vertical displacement.

### Solve 3 — Diffusion–Deformation Coupling Loop

Since the diffusion solve depends on k_eff(p) and the deformation solve
produces p, the two are iterated in a fixed-point loop:

```
repeat until Δp / p < tolerance:
    diffusion solve → C(x)           using current k_eff(p)
    deformation solve → v(x), p(x)   using current C(x)
    update k_eff(p) at each node
```

This feedback is the mechanism by which compressive stress at the mask edge
reduces oxidant consumption and limits bird's beak growth. Without this
coupling the bird's beak would be overestimated.

### Solve 4 — Mask Bending (Quasi-Static Linear Elasticity)

The nitride mask is modeled as a **viscous elastic solid** with temperature-
dependent creep viscosity (Arrhenius law). Inside the nitride domain, the
Lamé momentum equation is solved:

```
μ ∇²v_mask + (λ + μ) ∇(∇ · v_mask) = 0
```

where the Lamé viscosity parameters μ and λ are derived from the effective
Si₃N₄ creep viscosity `η(T)` and Poisson's ratio ν:

```
η(T) = η_ref · exp(E_a/R · (1/T − 1/T_ref))

μ   = η(T) / (2(1 + ν))
λ   = η(T) · ν / ((1 + ν)(1 − 2ν))
```

At the **contact interface** (mask nodes adjacent to the oxide), the oxide
full Cauchy stress tensor `σ_oxide = −p·I + s_dev` applies a traction on the
mask face. For a contact face with outward normal **n̂** (pointing from mask
into oxide):

```
t_i = Σ_j σ_oxide_ij · n̂_j
```

This traction is converted to a Neumann boundary condition for the mask
velocity, so the oxide compressive pressure directly bends the nitride.

**Physical interpretation of the mask bending solve:**

- Larger oxide pressure → larger traction → larger mask deflection.
- Stiffer mask (larger η_ref or lower temperature) → smaller deflection per
  unit traction → less bending.
- A thicker mask has more material between the contact face and the anchor at
  the far boundary, so the traction decays more before reaching the top — a
  thicker mask bends less.
- The contact is unilateral: the oxide can push the mask upward but cannot
  pull it downward. If the traction is tensile, no bending is applied on that face.

After solving, the mask velocity field V_mask(x) is fed back into the oxide
deformation solve as a Dirichlet boundary condition on the oxide/mask contact
faces (replacing the no-slip mask boundary). This closes the Solve 2 ↔ Solve 4
feedback loop: the oxide pushes the mask, the mask deflects, the deflected
mask position changes the oxide contact boundary, which changes the oxide
pressure distribution, which changes the oxide traction on the mask.

### The Constrained Ambient Velocity

Ambient-interface points that lie under the nitride (inside φ_mask < 0)
must **not** grow freely — they are constrained to move with the mask.
`OxidationConstrainedAmbientVelocityField` implements this:

```
if x is inside the mask:
    v_amb(x) = V_mask(x)    (moves with nitride, no free oxidation)
else:
    v_amb(x) = V_oxide(x)   (moves with free oxide surface)
```

This ensures that the pad oxide surface under the nitride tracks the mask
bottom face exactly, and that no fictitious oxide growth occurs under the mask.

### Mandatory Boolean Clips

Because the three level sets move independently between clip operations, the
ambient surface can slightly penetrate the nitride mask during an advection
step. Two boolean clips enforce non-penetration:

```
Before advection:  ambient = ambient \ mask    (pre-clip)
After advection:   ambient = ambient \ mask    (post-clip)
```

Both clips are structural — they always execute inside each time step.

---

## Per-Time-Step Workflow Summary

```
1. Diffusion + deformation coupled solve (Solves 1–3)
2. Mask bending solve (Solve 4)
3. Oxide/mask interface coupling iterations:
   - oxide solve with current mask velocity → updated oxide pressure
   - mask solve with updated oxide traction → updated mask velocity
   - repeat until contact-velocity change < tolerance (typically 3–8 iterations)
4. Pre-advection boolean clip: ambient = ambient \ mask
5. Advance φ_amb with constrained ambient velocity
6. Advance φ_Si with diffusion velocity
7. Advance φ_mask with mask bending velocity
8. Post-advection boolean clip: ambient = ambient \ mask
```

Steps 3–8 are what the `LOCOSOxidation` ViennaLS wrapper performs internally.
The ViennaPS `Oxidation` class calls this wrapper each time step.

---

## The ViennaPS `Oxidation` Model for LOCOS

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
model->setTimeStep(0.1);               // hr per LOCOS step
model->setMaskParameters(
    ls::OxidationMaterials<double>::siliconNitrideMask1000C());

// Run one time step at a time to save intermediate shapes
NumericType elapsed = 0., total = 1.0;
while (elapsed < total) {
    double dt = std::min(0.1, total - elapsed);
    model->setTime(dt);
    model->setTimeStep(dt);
    ps::Process<double, 2>(domain, model, 0.0).apply();
    elapsed += dt;
    domain->saveSurfaceMesh("locos_step_" + std::to_string(elapsed) + ".vtp");
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

elapsed = 0.0
while elapsed < 1.0:
    dt = min(0.1, 1.0 - elapsed)
    model.setTime(dt)
    model.setTimeStep(dt)
    vps.Process(domain, model, 0.0).apply()
    elapsed += dt
    domain.saveSurfaceMesh(f"locos_step_{elapsed:.2f}.vtp")
```

---

## Geometry Setup

```
Si substrate:         flat plane at y = 0
Pad SiO₂:            geometric offset of Si by padOxideThickness (0.15 µm)
Si₃N₄ mask:          box covering x < maskEdge (= 0), y from pad top to pad top + maskThickness (0.3 µm)
Open oxidation window: x > 0 (right half of domain)
```

The small contact epsilon (10⁻⁶ µm) between the mask bottom and the pad oxide
top ensures that Cartesian stencil edges unambiguously cross the mask boundary.
Without it, stencil nodes exactly at the mask bottom may be misclassified.

Domain:
```
x ∈ [−4, 4] µm   REFLECTIVE boundaries (mask continues to −∞ in practice)
y ∈ [−1, 2] µm   INFINITE boundaries
gridDelta = 0.05 µm
```

---

## Key Parameters

### Process parameters (`setXxx` on `ps::Oxidation`)

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | — | °C; sets k, D via Arrhenius fits |
| `oxidant` | — | `Wet` or `Dry`; sets pre-exponentials |
| `pressure` | 1.0 atm | Scales C* (equilibrium concentration) |
| `orientation` | `Si100` | Crystal anisotropy factor on k |
| `timeStep` | — | hr per LOCOS step; must satisfy CFL |
| `maxGridPoints` | 5×10⁶ | Limits memory for the Cartesian solve |

### Mask parameters (`setMaskParameters`)

The `OxidationMaskParameters` struct controls the Si₃N₄ mechanics:

| Parameter | Typical value | Physical meaning |
|---|---|---|
| `referenceViscosity` | 5×10¹¹ Pa·hr | Creep viscosity at `referenceTemperature` |
| `referenceTemperature` | 1273.15 K | Temperature where `referenceViscosity` applies |
| `creepActivationEnergy` | 0–600 kJ/mol | Arrhenius activation energy; 0 = temperature-independent |
| `poissonRatio` | 0.27 | Sets λ/μ ratio; controls volumetric vs. shear compliance |
| `unilateralContact` | true | Oxide pushes mask but cannot pull it |

`ls::OxidationMaterials<T>::siliconNitrideMask1000C()` returns calibrated
values for Si₃N₄ at 1000 °C that reproduce the bird's beak magnitude observed
in experiment.

A **stiffer mask** (larger `referenceViscosity` or larger `poissonRatio → 0.5`)
bends less for the same oxide pressure. This reduces the bird's beak extension
because the oxide cannot deflect the nitride to create additional volume for
lateral oxide growth. Experimentally, thicker nitride layers produce sharper
LOCOS profiles because the greater stiffness (per unit of contact-traction
response) limits bending.

### Mask coupling iterations

The oxide/mask interface coupling loop runs up to `maskCouplingIterations`
(default 8) times per LOCOS step and stops early when the relative change in
contact-face velocity falls below `maskCouplingTolerance` (default 2%). The
Aitken acceleration scheme typically achieves convergence in 3–7 iterations.
Increasing the iteration count does not improve accuracy if the tolerance is
already met; it only increases cost.

---

## What to Expect from the Simulation

After 1 hr of wet oxidation at 1000 °C, 1 atm:

- **Open window:** ~0.4–0.5 µm of oxide grows vertically; the oxide surface
  rises by ~0.28 µm (56% of total growth) and the Si interface sinks by
  ~0.22 µm (44% of total growth).
- **Bird's beak penetration:** The oxide extends ~0.2–0.4 µm under the mask
  edge, tapering smoothly to zero over ~0.5 µm.
- **Mask deflection:** The nitride bottom curves upward by ~0.05–0.1 µm at
  the mask edge, negligible at the anchor end.
- **Oxidant suppression under mask:** The oxidant concentration under the
  center of the mask is less than 0.1% of the open-window value — the
  nitride is an effective oxidant block.

The planar Deal-Grove estimate for the same conditions gives the open-window
oxide thickness to within ~5%, serving as a sanity check.

---

## Time-Stepping Considerations

LOCOS requires **multiple time steps** because:
1. The geometry changes substantially over 1 hr (the oxide thickness grows from
   padOxideThickness to ~0.5 µm), so a single large step would violate CFL.
2. Each LOCOS step re-solves the mask bending with the current oxide geometry;
   using many steps lets the mask shape evolve gradually and accurately.

Recommended: `timeStep = 0.05–0.1 hr`. Finer steps give marginally better
accuracy but proportionally higher cost. The included example uses
`timeStep = 0.1 hr` as a practical default.

The example saves a surface mesh after each step, allowing animation of the
bird's beak development in ParaView.

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
gridDelta=0.05
xExtent=4.0
yMin=-1.0       yMax=2.0
padOxideThickness=0.15
maskThickness=0.3
maskEdge=0.0
oxidationTime=1.0
timeStep=0.1
temperature=1000.
pressure=1.0
oxidant=wet
orientation=100
```

---

## Output Files

| File | Contents |
|---|---|
| `ps_locos_step_NNNN.vtp` | Surface mesh at each time step (all three materials) |
| `ps_locos_after.vtp` | Final surface mesh |

Open in ParaView. The three materials (Si, SiO₂, Si₃N₄) are labeled in the
domain metadata. Zoom into the mask edge region at x ≈ 0 to visualize the
bird's beak profile and the nitride bending.

---

## Connection to ViennaLS

`ps::Oxidation` uses the `LOCOSOxidation<T,D>` wrapper from ViennaLS when a
Si₃N₄ layer is present. For the implementation details of:
- The coupled diffusion + Stokes solver,
- The mask bending elasticity solve,
- The constrained ambient velocity field,
- The traction-driven contact node boundary condition,

see:

- `ViennaLS/examples/LOCOSOxidation/README.md` — full solver reference
- `lsLOCOSOxidation.hpp` — the LOCOS wrapper class
- `lsOxidationMask.hpp` — mask bending solver and constrained ambient

---

## Further Reading

- B.E. Deal and A.S. Grove, *General Relationship for the Thermal Oxidation
  of Silicon*, J. Appl. Phys. 36, 3770 (1965).
- K. Taniguchi, M. Tanaka, C. Hamaguchi, K. Imai, *Two-Dimensional Computer
  Analysis of LOCOS Process*, J. Electrochem. Soc. 137, 1589 (1990).
- P. Sutardja and W.G. Oldham, *Modeling of Stress Effects in Silicon
  Oxidation*, IEEE Trans. Electron Devices 36, 2415 (1989).
- E.A. Irene, *Silicon Oxidation Studies: A Quantitative Investigation of the
  Pressure Retardation of the Thermal Oxidation of Silicon*, J. Electrochem. Soc.
  125, 1708 (1978).
- D.B. Kao, J.R. McVittie, W.D. Nix, K.C. Saraswat, *Two-dimensional Thermal
  Oxidation of Silicon — I. Experiments*, IEEE Trans. Electron Devices 34, 1008
  (1987).
