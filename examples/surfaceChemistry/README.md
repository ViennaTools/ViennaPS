# Surface Chemistry from a Reaction File

One model, many chemistries. Deposition, etching, sputtering and a passivating
film are one code path here: **the reaction file decides which happens**, and the
sign of the surface velocity follows from the stoichiometry.

```bash
./surfaceChemistry reactions/silane.mechanism.json             # C++, deposition
./surfaceChemistry reactions/sf6o2.mechanism.json --mask 30    # C++, a masked etch
python surfaceChemistry.py reactions/diamond.yaml       # Python, same driver
```

Nothing about a particular chemistry is written in either driver. The file
decides whether the surface grows or is etched, which particles are traced, how
many coverages there are, what the rate laws are, and how the chemistry differs
from one material to the next.

## What is in this directory

| | |
|---|---|
| `surfaceChemistry.cpp` | the C++ driver: reads a mechanism file, builds the model, runs a trench or a hole |
| `surfaceChemistry.py` | the same in Python, reading either the `.yaml` or the mechanism data |
| `reactions/` | twelve reaction files, each with its compiled `.mechanism.json` |
| `demoMultiMaterial.py` | selective growth on a SiGe/Si superlattice |
| `demoPassivation.py` | a polymer film competing with an etch in a masked trench |
| `validation/diamondRadicalFraction.py` | the diamond mechanism against its published closed form |

## Running it

Both drivers take the same options:

```
-D, --dim 2|3        a trench (2D) or a cylindrical hole (3D)
    --thickness nm   how much film to grow, or how much to remove
    --mask nm        mask height; an etch needs one to show selectivity
    --width nm       feature width
    --depth nm       feature depth
    --grid nm        grid spacing
    --rays n         rays per surface point
    --gpu            trace on the device instead of the CPU
```

The Python driver adds `--time`, `--temperature`, `--material`, `--film` and
`--output`.

Each run first prints what the model *derived* from the file, before simulating:

```
mechanism   : diamond_cvd_standard_growth_model
temperature : 1200 K
solids      :  C (rho = 17.6 e22/cm3)
coverages   :  H*
particles   :  H_flux  H2_flux  CH3_flux
reactions   :
   H* + H -> H2 + *
   H2 + * -> H* + H
   H + * -> H*
   CH3 + * -> C + H* + H2
steady state:
   theta_H* = 8.483175e-01
   growth rate = 7.437752e-01 nm/s
```

Both write `<mechanism>_2D_initial/final.vtp` and a meshed volume
`..._volume.vtu`, which renders as a solid body in ParaView; colour it by
`Material` to see the film against the substrate.

## The two files beside each mechanism

`reactions/x.yaml` is what a user writes: species and their phases, the
reactions, and their rate constants. `reactions/x.mechanism.json` is the same
mechanism compiled by [ViennaChem](https://github.com/ViennaTools/ViennaChem):

```bash
python -m viennachem reactions/x.yaml reactions/x.mechanism.json
```

ViennaPS reads the compiled form in C++ (`psChemicalMechanismIO.hpp`), and that
is the *only* reader: the Python driver hands the same data to the same reader
through `ps.ChemicalMechanism.fromJSON`, so there is one implementation of the
format, not one per language. Every file carries a `schemaVersion`, and a reader
that does not know the version refuses the file rather than guessing.

## What each reaction file demonstrates

| file | what it shows | where the numbers come from |
|---|---|---|
| `silane.yaml` | the base case: LPCVD polysilicon from silane | checked against a by-hand implementation of the same mechanism |
| `silane_kads.yaml` | a sticking coefficient derived from an adsorption rate constant, `s = 4k_ads/v̄` | same chemistry, different input form |
| `silane_selective.yaml` | a chemistry restricted to one material | illustrative |
| `sige_stack.yaml` | per-material sticking *and* barrier on a SiGe/Si stack | illustrative; the Ge-catalysed H desorption is real |
| `gaas_reversible.yaml` | two site types (cation and anion) and a reversible step | Mountziaris & Jensen 1991, Table II, reduced to [S5], [S11], [S22] |
| `gaas_full.yaml` | the same paper's **complete** mechanism: 26 surface reactions, 7 coverages, two solids | Mountziaris & Jensen 1991, Table II, in full |
| `gaas_toy.yaml` | the smallest two-site mechanism, for the site-type tests | illustrative |
| `diamond.yaml` | a mechanism whose central step is **reversible** | Bristol CVD Diamond Group "standard growth model"; reproduces the published radical fraction to 0.26 % over 900–1400 K |
| `ar_sputter.yaml` | physical sputtering: an ion yield instead of a rate constant | illustrative; the threshold form is standard |
| `cf4ar_etch.yaml` | ion-enhanced etching, and ion–neutral synergy (15× the sum of the parts) | illustrative |
| `sf6o2.yaml` | **the acceptance test**: ViennaPS's own SF₆/O₂ silicon etch, written as a reaction file | every number from `SF6O2Etching::defaultParameters()`; matches the hand-written model to 0.16 % |
| `polymer_etch.yaml` | passivation competing with the etch: two solids, one deposited while the other is removed | illustrative; the forms are standard |

## Demonstrations beyond the driver

The driver grows or etches one mechanism in a trench or a hole. Three results
need more than that, so they are their own scripts — each still reading nothing
but a reaction file:

```bash
python demoMultiMaterial.py      # a SiGe/Si superlattice with a trench through it
python demoPassivation.py        # a masked trench with a polymer layer on top
python demoGaAsMechanism.py      # 26 reactions against 3, analytically (--trench too)
```

`demoMultiMaterial.py` exposes both materials side by side on the sidewall: the
film decorates the SiGe bands at 6.4× the Si rate, and the contrast decays from
6.38× to 1.19× as the growing film buries them.

`demoGaAsMechanism.py` asks what a reduced mechanism costs. At the paper's own
conditions the three-reaction reduction is within 0.007 % of all 26, and holds
that across 800-1300 K. Feed the surface the methyl radicals that TMG pyrolysis
releases, though, and [S24] strips the hydrogen off adsorbed AsH while [S26]
grows on the bare arsenic over a 20 kcal/mol barrier against [S22]'s 29.3 -- a
second, faster growth channel the reduction cannot see. At three times the
methylgallium flux it carries 4.3 % of the growth.

`demoPassivation.py` shows why a Bosch-type process works: the trench floor
faces the ion source, loses its film and etches ~27 nm, while the sidewalls see
almost no ion flux, keep their film and gain ~13 nm — in the same run. Nothing in
the model knows about sidewalls; the difference is the flux the ray tracer
delivers.

Both write `.vtp` and `_volume.vtu` files here.

## Validation

```bash
python validation/diamondRadicalFraction.py
```

Solves the diamond mechanism over 900–1400 K and compares the radical fraction
against the published closed form (0.26 % worst case). It runs no simulation: it
is a check of the framework rather than a demonstration of it.

## What this directory does and does not contain

**Running any of the twelve mechanisms needs nothing but this directory** and a
ViennaPS install. That holds for the C++ driver, the Python driver, both demos
and the validation script.

**Writing a twelfth needs one install.** Compiling a reaction file -- parsing the
equations, checking the atom balance, inferring the free sites, deriving the
stoichiometry -- is ViennaChem's job:

```bash
pip install git+https://github.com/ViennaTools/ViennaChem@main
```

Then the `.yaml` path works directly. Without it the driver still runs every
mechanism from its compiled form, says so rather than falling back silently, and
refuses outright if the `.yaml` is newer than the data beside it:

```
note: ViennaChem is not installed, so 'silane.mechanism.json' is used
      instead of the reaction file itself.
'silane.yaml' is NEWER than 'silane.mechanism.json': the mechanism data is
      stale. Recompile it with `python -m viennachem silane.yaml ...`.
```

The model itself is `include/viennaps/models/psSurfaceChemistry.hpp`, its file
reader `psChemicalMechanismIO.hpp`, and the device shader
`gpu/models/SurfaceChemistry.cuh`.
