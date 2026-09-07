# Diffusivity provenance for `annealing.csv`

Intrinsic Arrhenius diffusivities `D = D0 * exp(-Ea / kT)` used by the anneal
model. All values come from the open literature. The primary source is Pichler,
*Intrinsic Point Defects, Impurities, and Their Diffusion in Silicon*,
Springer-Verlag Wien (2004). Bismuth is from Ghoshtagore (1971), which Pichler
does not cover.

`D0` in `annealing.csv` is in nm^2/s (the ViennaPS length unit). Each entry lists
both nm^2/s and the original cm^2/s (nm^2/s = cm^2/s x 1e14). `D(1030 C)` is the
resulting diffusivity at 1030 C, for reference. Solid-solubility and effective-TED
references for boron and phosphorus are in the `annealing.csv` header.

## Boron (B, acceptor)
- D0 = 4.715e14 nm^2/s (4.715 cm^2/s), Ea = 3.674 eV, D(1030 C) = 2.918 nm^2/s
- Pichler (2004), Eq. (5.14), p. 346.
- Pichler's recommended intrinsic boron diffusivity is the Fermi-level expression
  Eq. (5.14): `D = 0.123*exp(-3.566/kT) + (p/ni)*4.21*exp(-3.671/kT) +
  (p/ni)^2*39.8*exp(-4.373/kT)` cm^2/s. At intrinsic conditions (p/ni = 1) it
  gives D(1030 C) = 2.918 nm^2/s, dominated (91%) by the singly-ionized (linear)
  term. D0 = 4.715 cm^2/s, Ea = 3.674 eV is an effective single-Arrhenius fit to
  that intrinsic curve over 900-1100 C (matches to < 0.1%). The equilibrium study
  of Christensen et al., APL 82, 2254 (2003) gives a somewhat higher
  0.06*exp(-3.12/kT) -> 5.15 nm^2/s.

## Phosphorus (P, donor)
- D0 = 1.03e14 nm^2/s (1.03 cm^2/s), Ea = 3.507 eV, D(1030 C) = 2.815 nm^2/s
- Pichler (2004), Eq. (5.26), p. 395.
- Pichler's regression over all intrinsic/inert data 900-1200 C (Ea 90% CI
  3.42-3.6 eV). The equilibrium study of Christensen et al., APL 82, 2254 (2003)
  gives 8e-4*exp(-2.74/kT) -> 2.02 nm^2/s (lower Ea, similar value at 1030 C).

## Aluminum (Al, acceptor)
- D0 = 3.17e13 nm^2/s (0.317 cm^2/s), Ea = 3.023 eV, D(1030 C) = 64.493 nm^2/s
- Pichler (2004), Eq. (5.19), p. 362.
- Pichler's regression over all intrinsic/inert data (Ea 90% CI 2.92-3.13 eV).
  Supersedes the single-paper Krause/Ryssel/Pichler JAP 91, 5645 (2002) and the
  older, lower Ghoshtagore (PRB 3, 2507, 1971) value.

## Gallium (Ga, acceptor)
- D0 = 3.81e14 nm^2/s (3.81 cm^2/s), Ea = 3.552 eV, D(1030 C) = 6.974 nm^2/s
- Pichler (2004), Eq. (5.21), p. 368.
- Pichler's regression over all intrinsic/inert data (Ea 90% CI 3.46-3.64 eV).
  Supersedes the older, lower Ghoshtagore (PRB 3, 2507, 1971) value.

## Indium (In, acceptor)
- D0 = 3.13e14 nm^2/s (3.13 cm^2/s), Ea = 3.668 eV, D(1030 C) = 2.039 nm^2/s
- Pichler (2004), Eq. (5.23), p. 375.
- Pichler's regression over all intrinsic/inert data (Ea 90% CI 3.47-3.87 eV).
  Supersedes the older, lower Ghoshtagore (PRB 3, 2507, 1971) value.

## Arsenic (As, donor)
- D0 = 8.85e14 nm^2/s (8.85 cm^2/s), Ea = 3.971 eV, D(1030 C) = 0.388 nm^2/s
- Pichler (2004), Eq. (5.30), p. 408.
- Pichler's regression over all intrinsic/inert data (Ea 90% CI 3.9-4.04 eV).
  Supersedes the older Ghoshtagore (PRB 3, 397, 1971) value.

## Antimony (Sb, donor)
- D0 = 4.09e15 nm^2/s (40.9 cm^2/s), Ea = 4.158 eV, D(1030 C) = 0.339 nm^2/s
- Pichler (2004), Eq. (5.34), p. 420.
- Pichler's regression over all intrinsic/inert data (Ea 90% CI 4.08-4.24 eV).
  Supersedes the older Ghoshtagore (PRB 3, 397, 1971) value.

## Bismuth (Bi, donor)
- D0 = 1.08e14 nm^2/s (1.08 cm^2/s), Ea = 3.85 eV, D(1030 C) = 0.139 nm^2/s
- Ghoshtagore, Phys. Rev. B 3, 397 (1971).
- Bismuth is not covered by Pichler (2004), so Ghoshtagore (1971) is the primary
  source used here. These are intrinsic, surface-effect-free values that run
  lower than the traditional textbook/TCAD (Fair) values.
