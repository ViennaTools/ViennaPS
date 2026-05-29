# Step Oxidation

Simulates thermal oxidation of a silicon step using the ViennaPS `Oxidation`
process model. The example demonstrates 2D (and optionally 3D) geometry,
covering planar-to-nonplanar oxidation, the volume-expansion split between the
two moving interfaces, and stress-dependent reaction-rate and diffusivity
feedback.

---

## Physical Background

### Why Two Moving Interfaces?

Thermal oxidation of silicon is described by the Deal-Grove model. Oxidant
(O₂ or H₂O) diffuses through the growing SiO₂ film and reacts at the buried
Si/SiO₂ interface. Because 0.44 nm of silicon is consumed for every 1 nm of
oxide grown, the oxidation advances **downward** into the silicon. Simultaneously,
the new oxide has 2.27× the volume of the silicon it replaces, so it must
expand **upward** through the SiO₂/ambient surface.

A geometry simulation therefore requires **two level sets**:

| Level set | Tracks | Direction of motion |
|---|---|---|
| φ_Si | Si/SiO₂ reaction interface | Inward into silicon (−y) |
| φ_amb | SiO₂/ambient free surface | Outward into gas (+y) |

For a planar film the fraction of total growth each interface carries is fixed
by the expansion coefficient γ = 2.27 (the SiO₂/Si molar volume ratio):

```
Si interface displacement:       1/γ       ≈ 0.44  of total oxide thickness
Ambient interface displacement: (γ−1)/γ    ≈ 0.56  of total oxide thickness
```

For a **non-planar geometry** (a step, a trench corner, a curved surface) the
split is still correct locally, but the two interfaces move in locally
different directions set by their respective level-set normals, and the oxide
deforms laterally to accommodate volume incompatibility. A purely 1D Deal-Grove
model cannot capture this — a 2D/3D PDE solution is needed.

### The Three-Stage Physics Model

At each time step the `Oxidation` model runs the following coupled solves
before advancing the level sets:

#### Stage 1 — Oxidant Diffusion (Deal-Grove PDE)

The oxide region is the Cartesian band defined by:

```
φ_Si(x) ≥ 0  AND  φ_amb(x) ≤ 0
```

Inside that band the steady-state diffusion equation is solved:

```
∇ · (D_eff ∇C) = 0
```

with embedded boundary conditions at every level-set crossing along the
Cartesian stencil edges:

- **At φ_Si (reaction boundary):**  
  `-D ∂C/∂n = k_eff · C`  
  Oxidant is consumed at rate k_eff. With large k_eff the concentration at the
  interface drops toward zero (reaction-limited regime).

- **At φ_amb (gas-transfer boundary):**  
  `-D ∂C/∂n = h · (C* − C)`  
  Oxidant enters from the gas at rate h. With large h (default) this becomes
  the Dirichlet condition C = C*, the equilibrium concentration.

The sub-grid boundary distances are computed from the level-set zero crossings
rather than by snapping to the nearest grid node, giving second-order accuracy
at curved interfaces.

The resulting concentration field C(x) is spatially varying — it reflects the
local oxide thickness and geometry. Points under a thick oxide far from the
gas surface see a lower concentration than points at a step corner where the
diffusion path is shorter. This is the core reason why **oxidation is
geometry-dependent** in 2D/3D.

#### Stage 2 — Oxide Deformation (Quasi-Static Stokes Flow)

Silicon-to-oxide conversion increases volume by a factor γ = 2.27. In a
planar geometry this volume is free to expand; on a non-planar surface the
surrounding geometry constrains it and **mechanical stress builds up**. The
model treats the oxide as a **viscous fluid** and solves the quasi-static
Stokes momentum equation inside the oxide band:

```
η ∇²v = ∇p − ∇ · s_dev
```

where:
- `v` is the oxide deformation velocity field
- `η` is the effective oxide viscosity (~10¹⁰ Pa·s at 1000 °C)
- `p` is the mechanical pressure
- `s_dev` is the Maxwell viscoelastic deviatoric stress tensor

Boundary conditions:
- **At φ_Si:** Dirichlet — velocity equals the local expansion speed  
  `v_Si = ((γ−1)/γ) · k_eff · C / N`  
  directed along the Si surface outward normal.
- **At φ_amb:** Traction-free — the stress at the free surface is balanced by ambient pressure (zero to first order).

The solution gives a **vector velocity field** V(x) throughout the oxide.
This is used to advect φ_amb. Because it is a proper Stokes solution, it
automatically captures lateral oxide flow at step corners — something that a
purely normal-speed advection cannot do.

#### Stage 3 — Pressure–Concentration Coupling

Compressive pressure in the oxide reduces the oxidation rate through an
Arrhenius-type factor (Sutardja and Oldham, 1988):

```
k_eff(p) = k · exp(−(p − p_ref) · V_k / (k_B · T))
```

The same mechanism applies to the diffusivity:

```
D_eff(p) = D · exp(−(p − p_ref) · V_D / (k_B · T))
```

Because the diffusion solve and the mechanics solve each depend on the output
of the other, they are iterated in a **coupling loop** until the relative
pressure change between iterations falls below a tolerance. With small activation
volumes (the default for Si at 1000 °C) this converges in a single pass;
larger values model LOCOS-style retardation under highly stressed oxide.

#### Level-Set Advection

After the coupled solve, each level set is advanced by the Hamilton-Jacobi
equation:

```
∂φ/∂t + v(x) |∇φ| = 0
```

- **φ_Si** is advected by the **scalar** velocity from the diffusion field:  
  `v_Si(x) = velocitySign · k_eff(x) · C(x) / (N · γ)`.  
  This is negative (φ_Si shrinks, silicon is consumed).

- **φ_amb** is advected by the **vector** velocity field from the Stokes
  deformation solver. Using a vector field (not just the normal component)
  correctly moves the free surface at corners and curved regions where lateral
  flow is significant.

The Engquist–Osher spatial scheme and forward-Euler temporal scheme are used
by default. Because the velocity depends on the current level-set geometry (via
the oxide thickness), the oxidation model automatically subcycles so the
coupled fields are recomputed before the interfaces move too far.

---

## The ViennaPS `Oxidation` Model

The `ps::Oxidation<T,D>` class orchestrates the entire workflow above. It:

1. Reads the Si and SiO₂ level sets from the ViennaPS domain.
2. Converts temperature and oxidant type to Deal-Grove rate constants using
   Arrhenius parameters (Massalski & Plummer fits, ~100 orientation).
3. Constructs an `OxidationDiffusion` velocity field and an
   `OxidationDeformation` velocity field (both from ViennaLS).
4. Wraps them in an `OxidationModel` coupling loop.
5. Advances the two level-set interfaces using CFL-limited internal substeps.
6. Repeats until the total `time` has elapsed.

The Si material is automatically identified by its `Material::Si` tag; SiO₂
is identified by `Material::SiO2`. If no SiO₂ layer exists, a thin native
oxide of `initialOxideThickness` is generated before the first step.

If a `Material::Si3N4` layer is present, LOCOS physics activate automatically
(see `locosOxidation/`).

### C++ API

```cpp
auto model = ps::SmartPointer<ps::Oxidation<double, 2>>::New();
model->setTemperature(1000.);          // °C
model->setTime(0.05);                   // hr
model->setTimeStep(0.01);               // hr; maximum internal oxidation step
model->setOxidant(ps::OxidantType::Wet);
model->setPressure(1.0);                // atm
model->setOrientation(ps::SiliconOrientation::Si100);
model->setInitialOxideThickness(0.);   // µm (0 = use existing SiO₂ layer)

ps::Process<double, 2>(domain, model, 0.0).apply();

// Diagnostic: expected planar thickness at this condition:
double expected = model->estimatePlanarOxideThickness(0.0);
```

### Python API

```python
import viennaps2d as vps

model = vps.Oxidation()
model.setTemperature(1000.)
model.setTime(0.05)
model.setTimeStep(0.01)
model.setOxidant(vps.OxidantType.Wet)
model.setPressure(1.0)
model.setOrientation(vps.SiliconOrientation.Si100)

vps.Process(domain, model, 0.0).apply()
print(model.estimatePlanarOxideThickness(0.0))
```

---

## Geometry

The silicon step is a 2D cross-section of a surface that has two heights:
- Flat Si surface at y = 0 for x < 0 (left side, lower)
- Raised Si surface at y = 1 µm for x > 0 (right side, higher)

The step wall at x = 0 is where lateral oxidation enhancement appears.

If `oxideThickness > 0`, a pre-existing oxide is constructed by geometrically
advancing the Si surface by that amount (an isotropic offset), which correctly
rounds the step corners.

```
Domain:   x ∈ [−4, 4] µm   REFLECTIVE boundaries (mirror symmetry)
          y ∈ [−2, 4] µm   INFINITE boundaries
gridDelta = 0.05 µm
```

The step at x = 0 produces a locally thin oxide in the corner region, which
diffuses more oxidant to the Si interface there and causes **corner
enhancement**: the oxide grows faster at inside corners and slower at outside
corners than on the flat surfaces.

---

## Key Parameters

### `setTemperature(T)` — °C, range 800–1200

Temperature drives the Deal-Grove Arrhenius rate constants:
```
B/A = (B/A)_ref · exp(−E_A/(k_B·T))     (linear rate constant)
B   = B_ref · exp(−E_B/(k_B·T))          (parabolic rate constant)
```
At 1000 °C wet oxidation of ⟨100⟩ Si gives B/A ≈ 0.74 µm/hr, B ≈ 0.314 µm²/hr.
At 1100 °C the rates roughly double. Lowering temperature extends the parabolic
regime (diffusion-limited) relative to the linear regime (reaction-limited).

### `setOxidant(OxidantType)` — Wet or Dry

- `Wet` (H₂O): faster growth; OH diffuses more easily through SiO₂ than O₂;
  roughly 10× higher B and B/A than dry at the same temperature.
- `Dry` (O₂): slower but higher-quality oxide with lower interface-state density.

In the model this changes the pre-exponential factors and activation energies
in the Deal-Grove fit.

### `setPressure(P)` — atm

Both B and B/A scale linearly with partial pressure of the oxidant. Higher
pressure increases the equilibrium concentration C* at the oxide surface,
raising both the diffusion flux and the reaction flux proportionally. This
can compensate for a lower temperature (see lecture slide 6).

### `setOrientation(SiliconOrientation)` — Si100, Si111, PolySi

Silicon oxidizes anisotropically: the (111) face oxidizes roughly 1.7× faster
than (100) in dry conditions, and about 1.4× in wet conditions. The model
applies an orientation correction factor:

```
k(n̂) = k · [1 + (r₁₁₁ − 1) · (1 − (n̂ · ê_100)²)]
```

where n̂ is the local Si surface normal. On a flat (100) wafer the factor is 1
everywhere. On a step edge the normal rotates toward (110) or (111) depending
on geometry, giving locally enhanced or reduced reaction rates.

### `setTimeStep(dt)` — hr

`dt` is a user cap on the internal oxidation step, not a forced advection
duration. Each internal step solves diffusion, deformation, and interface
coupling on the current geometry, then clamps the actual advection time to:

```
actual_dt ≤ min(dt, C_CFL · gridDelta / max_velocity, remaining_time)
```

If `dt` is larger than the CFL-limited value, the model automatically performs
multiple smaller internal substeps before returning. Smaller `dt` can still be
used to force more frequent physics updates.

### `setMaxGridPoints(N)` — integer

The diffusion and mechanics solves operate on a Cartesian grid spanning the
oxide narrow band. Memory scales as N × D (2D: bytes per grid point ~O(100)).
For a 4 µm × 4 µm domain at Δx = 0.05 µm the grid has ~6400 points in 2D,
well within the default limit of 5×10⁶. For 3D or finer grids, raise this
limit accordingly.

### Stress-coupling parameters

| Setter | Effect |
|---|---|
| `setReactionActivationVolume(V_k)` | Arrhenius pressure correction on k_eff; V_k ~ 10⁻³⁵ m³ for Si at 1000 °C |
| `setDiffusionActivationVolume(V_D)` | Pressure correction on D_eff; typically negligible |

Both are off (= 0) by default. They become significant for highly stressed
structures such as narrow LOCOS windows, where compressive oxide pressure
can reduce oxidation rates by 20–40%.

---

## Volume Conservation and the Deal-Grove Check

Run `estimatePlanarOxideThickness(x_i)` to get the analytical Deal-Grove
prediction for the expected oxide thickness after the elapsed time starting
from initial thickness x_i. For a flat surface the simulated result should
match this within a few percent, verifying that the diffusion solve, volume
split, and advection are all consistent.

For the step geometry, the corner regions deviate from the planar estimate:
- Inside corners grow faster (locally thinner diffusion path → more oxidant flux).
- Outside corners grow slower (laterally constrained oxide expansion builds pressure).

These deviations are physically correct and observable in experiment.

---

## Running the Example

### C++
```bash
cmake --build build --target stepOxidation
cd build/examples/stepOxidation && ./stepOxidation config.txt
```

### Python
```bash
python3 examples/stepOxidation/stepOxidation.py config.txt
```

Configuration keys in `config.txt` (lengths in µm, time in hours):
```
gridDelta=0.05
xExtent=4.0
yMin=-2.0   yMax=4.0
stepX=0.0
leftSiTop=0.0   rightSiTop=1.0
oxideThickness=0.0
oxidationTime=0.05
timeStep=0.01
temperature=1000.
pressure=1.0
oxidant=wet
orientation=100
```

---

## Output Files

| File | Contents |
|---|---|
| `ps_step_oxidation_initial.vtp` | Initial surface mesh (Si + SiO₂ if present) |
| `ps_step_oxidation_after.vtp` | Final surface mesh after oxidation |

View in ParaView. The two materials are distinguishable by their material tag.
The Si/SiO₂ boundary (the inner surface φ_Si) and the SiO₂/ambient surface
(the outer surface φ_amb) both appear in the output.

---

## Connection to ViennaLS

`ps::Oxidation` is a thin wrapper around the ViennaLS oxidation stack. For
the diffusion and deformation internals — including the embedded boundary
stencil assembly, the Stokes pressure-velocity solve, and the Maxwell
viscoelastic stress update — see the ViennaLS documentation:

- `ViennaLS/examples/StepOxidation/README.md` — full solver reference
- `lsOxidationDiffusion.hpp` — the diffusion field class
- `lsOxidationDeformation.hpp` — the Stokes mechanics class
- `lsOxidationModel.hpp` — the coupling loop

---

## Further Reading

- B.E. Deal and A.S. Grove, *General Relationship for the Thermal Oxidation
  of Silicon*, J. Appl. Phys. 36, 3770 (1965).
- P. Sutardja and W.G. Oldham, *Modeling of Stress Effects in Silicon
  Oxidation*, IEEE Trans. Electron Devices 36, 2415 (1989).
- H.-H. Tsai, C.-H. Pao, C.-T. Chen, *Crystal-Orientation Dependence of
  Si Oxidation Rate*, J. Electrochem. Soc. 143, 2323 (1996).
