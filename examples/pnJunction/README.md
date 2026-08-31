# pnJunction — Lateral PN Junction by Sequential Masked Implantation

This example simulates the formation of a **lateral PN junction** in silicon using
two sequential ion implantation steps with alternating masks, followed by a thermal
anneal.  It demonstrates how ViennaPS and ViennaCS work together to go from raw
implant moments to electrical characterisation (sheet resistance, junction position)
without ever calling an external process simulator.

---

## Table of Contents

1. [What Is a Lateral PN Junction?](#1-what-is-a-lateral-pn-junction)
2. [Device Geometry](#2-device-geometry)
3. [Physics of Ion Implantation](#3-physics-of-ion-implantation)
4. [The Dual-Pearson IV Distribution Model](#4-the-dual-pearson-iv-distribution-model)
5. [Lateral Straggle](#5-lateral-straggle)
6. [Solid Solubility and Electrical Activation](#6-solid-solubility-and-electrical-activation)
7. [Dopant Diffusion During Anneal](#7-dopant-diffusion-during-anneal)
8. [Simulation Steps — Step by Step](#8-simulation-steps--step-by-step)
9. [Calibration Strategy](#9-calibration-strategy)
10. [Configuration File Reference](#10-configuration-file-reference)
11. [Running the Example](#11-running-the-example)
12. [Output Files](#12-output-files)
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)
14. [References](#14-references)

---

## 1. What Is a Lateral PN Junction?

A **PN junction** is the boundary in a semiconductor where a region doped with
donors (n-type) meets a region doped with acceptors (p-type).  In a conventional
vertical junction (e.g., a diode diffusion), the junction lies at a certain
**depth** below the surface, and the dopants on either side come from a blanket
implant into the full wafer surface.

A **lateral junction** instead lies at a specific **x-position** in the plane of
the wafer.  The two doped regions sit side by side rather than one above the other.
This geometry is used in:

- **Bipolar base contacts** — where emitter and base regions are separated
  laterally
- **CMOS source/drain junctions** at the edge of the gate
- **Image sensor pixels** — where P and N wells are formed in adjacent regions
- **Junction isolation** in older bipolar processes

In this example the domain is split at **x = 0**: phosphorus (P, n-type) is
implanted into the left half (`x < 0`) and boron (B, p-type) into the right half
(`x > 0`).  The metallurgical junction — where the net doping changes sign —
nominally sits exactly at x = 0, but diffuses slightly during the anneal and is
broadened by **lateral straggle** of the ion beams near the mask edge.

---

## 2. Device Geometry

```
   x = -100 nm                x = 0                x = +100 nm
        │                      │                         │
        │   ◄── Air (20 nm) ──►│◄────── Air ─────────►   │  y = +40 nm
        ├───────────────────────┼─────────────────────────┤  y = +20 nm  (oxide top)
        │   SiO2 pad oxide      │   SiO2 pad oxide        │
        │      (20 nm)          │      (20 nm)            │
  ──────┼───────────────────────┼─────────────────────────┼──── y = 0   (Si surface)
        │                      │                         │
        │  P-implanted Si       │  B-implanted Si         │
        │  (n-type, left half)  │  (p-type, right half)   │
        │                      │                         │
  ──────┼───────────────────────┼─────────────────────────┼──── y = −150 nm
```

**Coordinate convention:** y increases upward; the Si/SiO2 interface is at y = 0,
and depth into the substrate is negative y.  x = 0 is the nominal junction boundary.

The 20 nm SiO2 pad oxide acts as a **screen** for both implants: the beam passes
through it without depositing, and the Pearson IV profile is applied starting at the
Si surface.  Dopant diffusion is blocked at the SiO2/Air interface during anneal.

| Parameter             | Value   | Set by config key      |
|-----------------------|---------|------------------------|
| Grid spacing (Δ)      | 2 nm    | `gridDelta`            |
| Domain width (x)      | 200 nm  | `xExtent`              |
| Pad oxide thickness   | 20 nm   | `padOxideThickness`    |
| Air above oxide       | 20 nm   | `topSpace`             |
| Si substrate depth    | 150 nm  | `substrateDepth`       |

The domain uses **reflective boundary conditions** in x (the left edge at
x = −100 nm acts as a mirror, as ViennaPS only simulates the half-space in y
for the level-set geometry) and **infinite boundary conditions** in y.

---

## 3. Physics of Ion Implantation

When a high-energy ion enters a crystalline solid it loses energy through two
mechanisms:

1. **Nuclear stopping** — elastic collisions with lattice atoms, displacing them
   from their sites and creating crystal damage (vacancies and interstitials).
   Dominant at low energies.
2. **Electronic stopping** — inelastic energy transfer to the electron cloud,
   dominant at high energies.  Does not displace atoms but slows the ion.

The ion eventually comes to rest at a depth called the **projected range** (Rp).
Because each ion travels a slightly different path, the stopping depth is
statistically distributed.  For a large population of identical ions the resulting
depth distribution is approximately a **Pearson IV** probability density function,
which generalises the Gaussian by allowing non-zero skewness and excess kurtosis:

```
f(y) ∝ exp[ −∫ (y − μ) / [a₀ + a₁y + a₂y²] dy ]
```

This four-moment family captures the asymmetric, non-Gaussian tails observed in
real implant profiles.

**Channeling** is a second physical effect that produces a long tail in the depth
profile.  If a fraction of ions enter the crystal aligned with a low-index crystal
axis or plane, the surrounding atoms guide them deep into the lattice with reduced
nuclear stopping.  These ions travel 2–5× further than the random-component ions
before stopping.  The channeling tail is captured by a second Pearson IV component
(the **Dual-Pearson IV** model).

---

## 4. The Dual-Pearson IV Distribution Model

The total as-implanted concentration at depth y is modelled as:

```
C_total(y) = dose × [f_head × f_PIV(y; μ_h, σ_h, β_h, γ_h)
                    + (1 − f_head) × f_PIV(y; μ_t, σ_t, β_t, γ_t)]
```

where:

| Parameter         | Physical meaning                   | Config key (P) | Config key (B) |
|-------------------|------------------------------------|----------------|----------------|
| `μ_h` (headMu)    | Projected range of random component | `pProjectedRange` | `bProjectedRange` |
| `σ_h` (headSigma) | Straggle (standard deviation) of random component | `pDepthSigma` | `bDepthSigma` |
| `β_h` (headBeta)  | Skewness — how far the tail extends toward the surface | `pSkewness` | `bSkewness` |
| `γ_h` (headGamma) | Excess kurtosis — peakedness relative to a Gaussian | `pKurtosis` | `bKurtosis` |
| `f_head`          | Fraction of dose in the random (head) component | `pHeadFraction` | `bHeadFraction` |
| `μ_t` (tailMu)    | Projected range of channeling component | `pTailProjectedRange` | `bTailProjectedRange` |
| `σ_t` (tailSigma) | Straggle of channeling component | `pTailDepthSigma` | `bTailDepthSigma` |
| `β_t` (tailBeta)  | Skewness of tail | `pTailSkewness` | `bTailSkewness` |
| `γ_t` (tailGamma) | Kurtosis of tail | `pTailKurtosis` | `bTailKurtosis` |

For P at 30 keV, 7° tilt, 23° rotation, 1×10¹⁴ cm⁻²: the head (random) component
carries 96.82% of the dose (f_head = 0.9682) and peaks at Rp ≈ 39.5 nm; the
channeling tail carries 3.18% and peaks around 103 nm.  For B at 10 keV, 7° tilt,
23° rotation, 1×10¹⁴ cm⁻², the head peaks at Rp ≈ 36.1 nm with 93.85% of dose.

The 7° tilt and 23° rotation are chosen together to randomise the beam direction
relative to all major low-index crystal axes and planes simultaneously.  7° breaks
axial channeling along ⟨100⟩; the 23° azimuthal rotation avoids major planar
channels (e.g., {110} planes at 0° and 45°).  The combination gives the most
reproducible, well-defined depth profile across production wafers.

---

## 5. Lateral Straggle

Ions do not travel in a perfectly straight line.  Nuclear collisions deflect them
sideways as well as slowing them, so the lateral position of each ion when it stops
is also statistically distributed.  This **lateral straggle** (σ_lat) is modelled
as a Gaussian spread of the local deposition profile in the x-direction.

**Critically, SIMS cannot measure lateral straggle.**  Secondary ion mass
spectrometry (SIMS) measures the depth profile averaged over a macroscopic beam
spot — it gives no information about how far ions spread sideways.  Lateral straggle
must come from one of:

1. **Binary Collision Approximation (BCA) Monte Carlo** (e.g., SRIM/TRIM) —
   the standard source.  The ViennaPS `modeldb` tables carry BCA-computed lateral
   straggle in the `headLateralSigma` and `tailLateralSigma` columns.  These values
   are physics-based and require no calibration experiment.

2. **2-D carrier profiling** (SSRM, cross-sectional SRP) — can directly measure
   the lateral transition width at a mask edge.  Used when sub-10 nm accuracy is
   needed.

3. **Device-performance calibration** — infer σ_lat from C(V) measurements,
   junction capacitance, or threshold voltage roll-off in MOSFETs.

This example uses BCA values from the modeldb tables.  The practical impact:

| Species | depthSigma | headLateralSigma (BCA) | 0.3×σ rule of thumb |
|---------|-----------|------------------------|---------------------|
| P 30 keV, 7° tilt | 25.1 nm | **17.0 nm** (constant in table) | 7.5 nm (2.3× too small) |
| B 10 keV, 7° tilt | 21.2 nm | **24.0 nm** head, **27.0 nm** tail | 6.4 nm (4× too small) |

Using the rule of thumb would make the simulated junction appear twice as abrupt
as in reality.  The 10%–90% lateral transition width is approximately
2.56 × σ_lat, so for P this is about 44 nm — a significant fraction of the
100 nm half-width of the implant opening.

---

## 6. Solid Solubility and Electrical Activation

Immediately after implantation, dopant atoms sit primarily on **interstitial**
sites in the crystal lattice — they do not contribute to electrical conduction.
Only dopants on **substitutional** lattice sites (replacing a Si atom) are
electrically active.

The maximum concentration that can be substitutionally activated at a given
temperature is the **solid solubility**, C_ss(T).  It follows an Arrhenius form:

```
C_ss(T) = C₀ × exp(−Ea / kT)
```

where T is temperature in Kelvin, k is Boltzmann's constant, and C₀ and Ea are
species-specific parameters.

| Species | C₀ (nm⁻³) | Ea (eV) | C_ss at 1000 °C    | Config keys |
|---------|-----------|---------|---------------------|-------------|
| P in Si | 42.0      | 0.45    | ≈0.074 nm⁻³ = 7.4×10¹⁹ cm⁻³ | `pSolidSolubilityC0`, `pSolidSolubilityEa` |
| B in Si | 65.0      | 0.59    | ≈0.30 nm⁻³ = 3.0×10²⁰ cm⁻³  | `bSolidSolubilityC0`, `bSolidSolubilityEa` |

**Step 3** (zero-time activation) computes the active concentration field by
clamping the total implanted concentration to C_ss(T) cell by cell.  This is
equivalent to the Sentaurus command `diffuse time=0`, which activates without
diffusing.  Regions where C_total < C_ss are fully activated; regions above C_ss
are clamped.  In this example at 30 keV / 1×10¹⁴ cm⁻², P peaks at roughly
0.05 nm⁻³ — below the 1000 °C solubility limit of 0.074 nm⁻³ — so essentially
all P is activated.  B at 10 keV also peaks below its 0.30 nm⁻³ limit.

---

## 7. Dopant Diffusion During Anneal

During the thermal anneal, electrically active dopants diffuse by a thermally
activated random walk.  The diffusion coefficient follows:

```
D(T) = D₀ × exp(−Ea / kT)    [nm²/s]
```

The active concentration profile C_active(y, t) evolves according to Fick's
second law:

```
∂C_active/∂t = ∇ · [D(T) ∇C_active]
```

The ViennaCS `Anneal` model solves this PDE on the cell set using an implicit
finite-difference scheme, which is unconditionally stable for large time steps.

| Species | D₀ (nm²/s) | Ea (eV) | D at 1000 °C        | Config keys |
|---------|------------|---------|---------------------|-------------|
| P in Si | 4.0×10¹³  | 4.1     | ≈4.8×10⁻³ nm²/s     | `pAnnealD0`, `pAnnealEa` |
| B in Si | 7.57×10¹³ | 3.46    | ≈1.7×10⁻² nm²/s     | `bAnnealD0`, `bAnnealEa` |

The diffusion length after 30 s is √(2Dt):

```
P: √(2 × 4.8×10⁻³ × 30) ≈ 0.54 nm
B: √(2 × 1.7×10⁻² × 30) ≈ 1.0 nm
```

At these conditions the profiles move very little — the anneal is primarily
activating defects rather than redistributing dopants.  At longer times or higher
temperatures the profiles would broaden and the junction position would shift.

**Note on Fermi-level enhancement:** In reality, at high doping levels the
diffusivity of each species depends on the local carrier concentration through the
Fermi level.  This coupling between P and B is not implemented in this example
(it is Step 2 on the ViennaCS roadmap).  For the well-separated geometry here
(P and B are implanted into separate halves), the error is small except very near
the junction boundary where both concentrations are comparable.

---

## 8. Simulation Steps — Step by Step

### Step 1 & 2: Sequential Masked Implantation

The simulation uses a single shared domain throughout.  Instead of physically
creating a mask geometry (which would require building a separate level-set
surface), the example temporarily **overwrites the Material field** of Air cells
on the blocked half to `Mask`, runs the implant, then restores the original
Material values.

This works because `IonImplantation` respects the `setMaskMaterials()` list when
depositing dose — cells tagged as Mask absorb the beam without accumulating
concentration.  The cell set already contains a `Material` scalar field (one value
per cell) that governs this.

```
Before P implant:  Air | Air | Air | Air | ... | Air
After  mask_half:  Air | Air | Air | Mask| ... | Mask    (right half blocked)
After  P implant:  P_total deposited only in left half
After  restore:    Air | Air | Air | Air | ... | Air     (domain unchanged)
```

Both `P_total` and `B_total` accumulate in the **same cell set**, so a single
anneal step can diffuse both species simultaneously using the same underlying
concentration grid.

### Step 3: Zero-Time Activation

```cpp
annealP->applyActivation(domain);   // writes P_active = min(P_total, C_ss(T))
annealB->applyActivation(domain);   // writes B_active = min(B_total, C_ss(T))
```

This initialises the `P_active` and `B_active` fields that subsequent steps use.
Without this step, `NetDoping` and `SheetResistance` would see zero or uninitialised
concentrations.

### Step 4: Thermal Anneal

Each species is annealed independently:

```cpp
Process<T,D>(domain, annealP, 0.).apply();   // diffuses P_active in Si
Process<T,D>(domain, annealB, 0.).apply();   // diffuses B_active in Si
```

The `Process` call passes the anneal model to the ViennaPS process engine.  The
`Anneal` model confines diffusion to `diffusionMaterials = {Si}` and treats
`blockingMaterials = {Air}` as impenetrable surfaces — so dopants cannot diffuse
out of the silicon surface.

### Step 5: NetDoping and Junction Location

`NetDoping::apply()` computes:

```
net_doping(cell) = Σ(donor labels) − Σ(acceptor labels)
                 = P_active − B_active
```

A positive value means n-type (P dominates); negative means p-type (B dominates).
The metallurgical junction is where `net_doping = 0`.

`lateralJunctionPosition(y)` scans the `net_doping` field along x at a fixed depth
y and returns the x-coordinate(s) where it crosses zero by linear interpolation.
This is a direct implementation of the standard definition of the junction position
in 2-D device simulation.

```
net_doping(x) at y = -50 nm:

 +0.05 nm⁻³  ┤ ██████████                  (P side)
              │           ██
              │             ██
 0            ┼───────────────▏ ← x_j       (junction)
              │               ██
              │                 ██
 -0.05 nm⁻³  ┤                   ████████   (B side)
              ─────────────────────────── x
           -100 nm      0       +100 nm
```

### Step 6: Sheet Resistance

Sheet resistance (Rsh, units Ω/□) is a key electrical figure of merit that
integrates the conductivity profile through the full junction depth:

```
Rsh = 1 / (q × ∫ μ(y) × C_active(y) dy)
```

where μ(y) is the carrier mobility (which depends on local doping concentration
via the Caughey-Thomas model) and q is the elementary charge.  ViennaCS
`SheetResistance` evaluates this integral numerically over all cells with nonzero
concentration.

`computeElectron()` uses electron mobility and is appropriate for the n-side (P).
`computeHole()` uses hole mobility and is appropriate for the p-side (B).

**Caveat:** Because the integrals span the whole domain (both x-halves), the
reported Rsh values include contributions from both sides.  For a true one-sided
Rsh, restrict the sum to x < 0 (n-side) or x > 0 (p-side) by filtering cells.

---

## 9. Calibration Strategy

Each parameter class has a distinct experimental source.  Substituting calibrated
values requires only editing `config.txt`.

### Vertical Pearson IV Moments

**Source: SIMS on a blanket-wafer implant.**

Perform a 1-D implant into a bare Si wafer (no mask), measure the depth profile by
SIMS, and fit a Dual-Pearson IV to the data.  ViennaPS provides
`SimulationCalibrator` (see the `ionImplantation` example) to automate this.

Parameters calibrated this way: `pProjectedRange`, `pDepthSigma`, `pSkewness`,
`pKurtosis`, `pHeadFraction`, and the corresponding tail parameters.

These moments are **geometry-independent**: they describe the depth distribution of
a single ion entering a given material stack.  Calibrated values from a blanket
wafer apply identically to the masked geometry, because the mask simply gates which
surface cells receive dose — the Pearson IV shape within each open cell is unchanged.

### Lateral Straggle

**Source: BCA (SRIM Monte Carlo) — already in `config.txt`.**

The `modeldb` tables include `headLateralSigma` and `tailLateralSigma` columns
pre-computed by SRIM for each energy/tilt/dose combination.  This example reads
those values directly from the table, bypasses them into `config.txt`, and never
calls the modeldb at runtime.

For sub-10 nm accuracy at mask edges, calibrate from:
- **SSRM** (Scanning Spreading Resistance Microscopy): gives a 2-D carrier map,
  from which the lateral transition width is directly measurable.
- **C(V) measurements**: junction capacitance vs. voltage probes the effective
  depletion width, which depends on the lateral abruptness.

### Anneal Diffusivity (D₀, Ea)

**Source: Post-anneal SIMS on a blanket wafer.**

Perform the same anneal on a blanket-implanted sample, measure the SIMS profile
before and after, and fit the diffusion equation.  `AnnealCalibrator` in ViennaCS
automates the optimisation.

### Solid Solubility (C₀, Ea)

**Source: Sheet resistance measurement on a blanket annealed sample.**

The solid solubility limit sets the maximum active concentration.  `AnnealCalibrator`
with the `--rsh` flag optimises solubility parameters by minimising the error
between simulated and measured Rsh.  Alternatively, literature values (Trumbore
1960; Solmi et al. 1996) can be used as starting points — these are what the
current `config.txt` contains.

### Summary Table

| Parameter class            | Source                       | Tool / method                        |
|----------------------------|------------------------------|--------------------------------------|
| Vertical moments (Rp, σ, β, γ) | SIMS (blanket wafer)    | `SimulationCalibrator`               |
| Head fraction              | SIMS (channeling tail)       | `PearsonIVFitter` / `SimulationCalibrator` |
| Lateral straggle           | BCA modeldb table            | Read from CSV; in `config.txt`       |
| Anneal D₀, Ea              | Post-anneal SIMS             | `AnnealCalibrator`                   |
| Solid solubility C₀, Ea    | Rsh measurement              | `AnnealCalibrator --rsh`             |

---

## 10. Configuration File Reference

All parameters live in `config.txt` (copied to the build directory by CMake).
Lines beginning with `#` are comments.  The format is `key=value`.

### Geometry and process conditions

| Key                   | Value    | Unit | Description |
|-----------------------|----------|------|-------------|
| `gridDelta`           | 2.0      | nm   | Cell size (grid spacing) |
| `xExtent`             | 200.0    | nm   | Total domain width; P/B boundary at x = 0 |
| `padOxideThickness`   | 20.0     | nm   | SiO2 pad oxide between Si surface and air; acts as implant screen |
| `topSpace`            | 20.0     | nm   | Air layer above the pad oxide |
| `substrateDepth`      | 150.0    | nm   | Si depth below surface |
| `annealTemperatureC`  | 1000.0   | °C   | Anneal temperature (converted to K internally) |
| `annealTimeS`         | 30.0     | s    | Anneal duration |
| `junctionScanDepthNm` | 37.0     | nm   | Depth at which to scan for the lateral junction (P peak ≈ 39.5 nm, B peak ≈ 36.1 nm) |

### Phosphorus implant — head component (random)

Bilinear interpolation from the P/Si crystalline modeldb table:
tilt ∈ {6°, 10°} → 7° (t = 0.25),  rotation ∈ {22°, 45°} → 23° (t = 0.043).

| Key                 | Value    | Unit  | Description |
|---------------------|----------|-------|-------------|
| `pDoseCm2`          | 1.0e14   | cm⁻²  | Implant dose |
| `pTiltDeg`          | 7.0      | °     | Beam tilt from surface normal |
| `pRotationDeg`      | 23.0     | °     | Azimuthal (twist) angle; encoded in moments, no API setter |
| `pProjectedRange`   | 39.47    | nm    | Mean projected range (Rp) |
| `pDepthSigma`       | 25.10    | nm    | Depth straggle (σ) |
| `pSkewness`         | 5.7828   | —     | Skewness (β) |
| `pKurtosis`         | 1.0834   | —     | Excess kurtosis (γ) |
| `pLateralSigmaHead` | 17.0     | nm    | Lateral straggle — BCA table value; constant across all P 30 keV tilt/rotation entries |

### Phosphorus implant — tail component (channeling)

| Key                   | Value    | Unit  | Description |
|-----------------------|----------|-------|-------------|
| `pHeadFraction`       | 0.9682   | —     | Fraction of dose in the random (head) component |
| `pTailProjectedRange` | 103.26   | nm    | Channeling-tail projected range |
| `pTailDepthSigma`     | 70.00    | nm    | Channeling-tail depth straggle |
| `pTailSkewness`       | 3.3412   | —     | Tail skewness |
| `pTailKurtosis`       | 0.5608   | —     | Tail kurtosis |
| `pTailLateralSigma`   | 15.6     | nm    | Tail lateral straggle — constant across all P 30 keV entries |

### Phosphorus anneal physics

Source: `modeldb/anneal/annealing.csv`, species = phosphorus, material = silicon.

| Key                  | Value  | Unit  | Description |
|----------------------|--------|-------|-------------|
| `pAnnealD0`          | 4.0e13 | nm²/s | Arrhenius diffusivity pre-exponential factor |
| `pAnnealEa`          | 4.1    | eV    | Diffusion activation energy |
| `pSolidSolubilityC0` | 42.0   | nm⁻³  | Solid-solubility pre-exponential factor (≈ 4.2×10²² cm⁻³) |
| `pSolidSolubilityEa` | 0.45   | eV    | Solid-solubility activation energy; C_ss(1000 °C) ≈ 7.4×10¹⁹ cm⁻³ |

### Boron implant — head component (random)

Direct read from the B/Si crystalline modeldb table at tilt = 7°;
linear interpolation in rotation between 22° and 30° (t = 0.125).

| Key                 | Value    | Unit  | Description |
|---------------------|----------|-------|-------------|
| `bDoseCm2`          | 1.0e14   | cm⁻²  | Implant dose |
| `bTiltDeg`          | 7.0      | °     | Beam tilt from surface normal |
| `bRotationDeg`      | 23.0     | °     | Azimuthal (twist) angle; encoded in moments, no API setter |
| `bProjectedRange`   | 36.08    | nm    | Mean projected range (Rp) |
| `bDepthSigma`       | 21.19    | nm    | Depth straggle (σ) |
| `bSkewness`         | 3.2900   | —     | Skewness (β) |
| `bKurtosis`         | 0.5090   | —     | Excess kurtosis (γ) |
| `bLateralSigmaHead` | 24.0     | nm    | Lateral straggle — BCA table value (constant across all rotation entries at tilt = 7°) |

### Boron implant — tail component (channeling)

| Key                   | Value    | Unit  | Description |
|-----------------------|----------|-------|-------------|
| `bHeadFraction`       | 0.9385   | —     | Fraction of dose in the random (head) component |
| `bTailProjectedRange` | 68.30    | nm    | Channeling-tail projected range |
| `bTailDepthSigma`     | 46.10    | nm    | Channeling-tail depth straggle |
| `bTailSkewness`       | 3.5700   | —     | Tail skewness |
| `bTailKurtosis`       | −0.4590  | —     | Tail kurtosis |
| `bTailLateralSigma`   | 27.0     | nm    | Tail lateral straggle — BCA table value; constant across rotation entries |

### Boron anneal physics

Source: Fahey et al. 1989 (diffusivity); `modeldb/anneal/annealing.csv` (solid solubility).

| Key                  | Value    | Unit  | Description |
|----------------------|----------|-------|-------------|
| `bAnnealD0`          | 7.57e13  | nm²/s | Arrhenius diffusivity pre-exponential factor |
| `bAnnealEa`          | 3.46     | eV    | Diffusion activation energy |
| `bSolidSolubilityC0` | 65.0     | nm⁻³  | Solid-solubility pre-exponential factor (≈ 6.5×10²² cm⁻³) |
| `bSolidSolubilityEa` | 0.59     | eV    | Solid-solubility activation energy; C_ss(1000 °C) ≈ 3.0×10²⁰ cm⁻³ |

---

## 11. Running the Example

### C++ (CMake build)

```bash
cd /path/to/ViennaPS/build
cmake ..
make pnJunction
cd artifacts   # or wherever CMake places the binary
./pnJunction                    # reads config.txt in the current directory
./pnJunction /path/to/config.txt  # explicit config path
```

CMake copies `config.txt` from the source directory to the build directory, so
the executable finds it automatically in the build artifact directory.

### Python

```bash
cd examples/pnJunction
python pnJunction.py                        # reads config.txt in CWD
python pnJunction.py path/to/config.txt     # explicit config
python pnJunction.py --no-vtk               # skip VTK output
python pnJunction.py --plot                 # show matplotlib plots
```

Requires ViennaPS Python bindings (`pip install viennaps` or build with
`-DVIENNAPS_BUILD_PYTHON=ON`).

---

## 12. Output Files

| File                             | Format | Contents |
|----------------------------------|--------|----------|
| `pnJunction_P_depth.csv`         | CSV    | P_active peak concentration vs. depth (nm, nm⁻³) |
| `pnJunction_B_depth.csv`         | CSV    | B_active peak concentration vs. depth |
| `pnJunction_netdoping_depth.csv` | CSV    | net_doping (P_active − B_active) peak vs. depth |
| `pnJunction_lateral.csv`         | CSV    | net_doping vs. x at the scan depth |
| `pnJunction_cellset.vtu`         | VTK    | Full 2-D cell set with all scalar fields (C++ output: `pnJunction_net_depth.csv`) |
| `pnJunction.png`                 | PNG    | Three-panel plot (only with `--plot`) |

### Visualising with ParaView

Open `pnJunction_cellset.vtu` in [ParaView](https://www.paraview.org/).  The
cell set stores `Material`, `P_total`, `P_damage`, `B_total`, `B_damage`,
`P_active`, `B_active`, and `net_doping` as scalar fields.  Use the
**Color by** dropdown to switch between fields.  A diverging colour map (blue–white–red)
works well for `net_doping` to show n-type and p-type regions.

### Interpreting the Lateral Profile

`pnJunction_lateral.csv` gives the net doping as a function of x at the scan
depth.  The sign change from positive (n, P-dominated) to negative (p, B-dominated)
marks the metallurgical junction.  The width of the transition zone is approximately
`2.56 × σ_lat` (the 10%–90% distance for a Gaussian transition), reflecting the
lateral straggle of both species at the mask edge.

---

## 13. Known Limitations and Future Work

### No Fermi-Level Coupling

The diffusion of P and B is computed independently.  In reality, at high doping
levels the electron concentration shifts the Fermi level, which in turn changes the
concentration of positively/negatively charged point defects and thereby the
effective diffusivity of each species.  This enhancement can be a factor of 2–5
in heavily doped regions.  The `pImplantManual` example includes defect-coupled
diffusion (SCORE model); extending this to two simultaneous species is Step 2 on
the ViennaCS roadmap.

### No Defect-Mediated Diffusion

This example uses the simple Arrhenius model (`setArrheniusParameters`), which
treats D as constant at the anneal temperature.  The full SCORE model, used in
`pImplantManual`, couples diffusion to the local point-defect (interstitial +
vacancy) concentrations generated by the implant.  This is important for
accurately predicting transient-enhanced diffusion (TED) during the early part of
the anneal.

### Mask Is Not a Physical Structure

The masking is implemented by temporarily overwriting the `Material` field rather
than by inserting a physical mask layer into the level-set domain.  This means:
- There is no photoresist thickness effect on lateral shadowing.
- The mask is perfectly abrupt (step function at x = 0).
- Tilt-induced shadow effects from a tall mask are not modelled.

For more realistic mask modelling, use `lsMakeGeometry` to build an actual
`Material::Mask` box and insert it as a level-set layer, as done in the
`pImplantManual` example.

### Lateral Sigma Source: Always Use Tilt-Matched Table Rows

The modeldb `headLateralSigma` and `tailLateralSigma` columns vary significantly
with tilt angle.  For B at 10 keV, the tail lateral sigma is 27 nm at 7° tilt
vs. 12 nm at 0° tilt — a factor of 2.25× difference.  Always match the tilt (and
rotation) when selecting the interpolation brackets; using 0°-tilt rows for a 7°
implant will substantially underestimate the lateral spread.

### 2-D Only

The example is compiled and run in 2-D (`D = 2`).  The full 3-D case requires
a 3-D domain and significantly more memory and compute time, but the code structure
is identical — change `constexpr int D = 2` to `D = 3` and rebuild.

---

## 14. References

1. **Pearson IV distribution for implant profiles:**
   Tasch, A.F. et al., "An improved approach to accurately model the as-implanted
   B and BF₂ profiles in silicon," *Journal of the Electrochemical Society*,
   136(3), 810–814 (1989).

2. **Dual-Pearson IV for channeling tails:**
   Hobler, G. & Selberherr, S., "Monte Carlo simulation of ion implantation into
   two- and three-dimensional structures," *IEEE Transactions on Computer-Aided
   Design*, 7(2), 174–180 (1988).

3. **Phosphorus diffusivity in Si (D₀ = 4×10¹³ nm²/s, Ea = 4.1 eV):**
   ViennaPS `modeldb/anneal/annealing.csv`, species=phosphorus, material=silicon.

4. **Boron diffusivity in Si (D₀ = 7.57×10¹³ nm²/s, Ea = 3.46 eV):**
   Fahey, P.M., Griffin, P.B. & Plummer, J.D., "Point defects and dopant
   diffusion in silicon," *Reviews of Modern Physics*, 61(2), 289 (1989).

5. **Solid solubility of P and B in Si:**
   Trumbore, F.A., "Solid solubilities of impurity elements in germanium and
   silicon," *Bell System Technical Journal*, 39(1), 205–233 (1960).
   Also: Solmi, S. et al., "High-concentration boron diffusion in silicon,"
   *Journal of Applied Physics*, 68(7), 3250 (1990).

6. **BCA lateral straggle:**
   Ziegler, J.F., Ziegler, M.D. & Biersack, J.P., "SRIM – The stopping and
   range of ions in matter," *Nuclear Instruments and Methods in Physics Research B*,
   268(11–12), 1818–1823 (2010).

7. **Sheet resistance integration (Caughey-Thomas mobility model):**
   Caughey, D.M. & Thomas, R.E., "Carrier mobilities in silicon empirically
   related to doping and field," *Proceedings of the IEEE*, 55(12), 2192–2193
   (1967).
