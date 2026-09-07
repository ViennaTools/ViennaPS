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

## Contents

| | |
|---|---|
| `surfaceChemistry.cpp` | the C++ driver: reads a mechanism file, builds the model, runs a trench or a hole |
| `surfaceChemistry.py` | the same in Python, reading either the `.yaml` or the mechanism data |
| `reactions/` | nineteen reaction files, each with its compiled `.mechanism.json` |
| `cyclicProcess.cpp` | a cyclic process: two chemistries, coverages carried from phase to phase |
| `cyclicProcess.py` | the same in Python, trench or lateral cavity alike |
| `reactions/sin_peald_cycle.py` | the same cycle integrated in time with no geometry: the place to fit the ALD rate constants |
| `demoMultiMaterial.py` | selective growth on a SiGe/Si superlattice |
| `demoPassivation.py` | a polymer film competing with an etch in a masked trench |
| `validation/testsuite.py` | the reference test run: every mechanism solved at a point, against stored output |
| `validation/diamondRadicalFraction.py` | the diamond mechanism against its published closed form |

## Running it

Both drivers take the same options:

```
-D, --dim 2|3        a trench (2D) or a cylindrical hole (3D)
    --thickness nm   how much film to grow, or how much to remove
    --mask nm        mask height; an etch needs one to show selectivity
    --width nm       feature width, or fin width with --fin
    --depth nm       feature depth, or fin height with --fin
    --fin            a fin instead of a trench, so the corner is convex
    --stack n        a Si/SiGe superlattice of n pairs instead of a substrate
    --layer nm       thickness of one layer of the stack
    --oxide nm       oxide cap above the stack
    --taper deg      mask sidewall taper, and the trench taper with --stack
    --extent nm      lateral domain width
    --grid nm        grid spacing
    --rays n         rays per surface point
    --time s         run for a time instead of removing a thickness
    --temperature K  override the temperature the file declares
    --flux LABEL=v   override an incident flux; 'ion' reaches the ion source
    --sticking L=f   multiply a species' sticking, rate law and rays together
    --rate I=v       set reaction I's prefactor, by the index the run prints
    --ion-energy eV  mean ion energy, keeping the relative spread
    --handwritten    run ViennaPS's own SF6O2Etching class instead of the file
    --intermediate   write the flux, coverage and velocity fields each step
    --bench N        time the coverage solve alone, over N surface points
    --profile        report the coverage solve as a fraction of the whole run
    --height nm      fix the vertical domain extent instead of deriving it
    --film name      material label for the deposited film, e.g. PolySi
    --snapshots n    write the surface n times instead of only at the end
    --out stem       output file stem
    --gpu            trace on the device: lines in 2D, triangles in 3D
```

The Python driver adds `--time`, `--temperature`, `--material`, `--film` and
`--output`.

`--film` names the material the deposit is labelled with, which is what
per-material rate entries are matched against and what ParaView colours by.
A mechanism that both deposits and etches needs it even though its net rate is
negative, because the film has to exist before a `materials:` entry can name
it.

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
format, one shared by both languages. Every file carries a `schemaVersion`, and
a reader refuses any version it recognises as later than its own.

## The capability each reaction file demonstrates

| file | what it shows | where the numbers come from |
|---|---|---|
| `silane.yaml` | the base case: LPCVD polysilicon from silane | checked against a by-hand implementation of the same mechanism |
| `silane_inputs.yaml` | both alternative input forms at once: the SiH₄ supply as a partial pressure (mTorr), `Γ = p/√(2πmk_BT)`, and the adsorption as a rate constant in cm/s, `s = 4k_ads/v̄` | same chemistry, same answer to 0.008 % |
| `silane_selective.yaml` | a chemistry restricted to one material | illustrative |
| `sige_stack.yaml` | per-material sticking *and* barrier on a SiGe/Si stack | illustrative; the Ge-catalysed H desorption is real |
| `gaas_reversible.yaml` | two site types (cation and anion) and a reversible step | Mountziaris & Jensen 1991, Table II, reduced to [S5], [S11], [S22] |
| `gaas_cvd.yaml` | the same paper's **complete** mechanism: 26 surface reactions, 7 coverages, two solids | Mountziaris & Jensen 1991, Table II, in full |
| `gaas_toy.yaml` | the smallest two-site mechanism, for the site-type tests | illustrative |
| `diamond.yaml` | a mechanism whose central step is **reversible** | Bristol CVD Diamond Group "standard growth model"; reproduces the published radical fraction to 0.26 % over 900–1400 K |
| `ar_sputter.yaml` | physical sputtering: an ion yield instead of a rate constant | illustrative; the threshold form is standard |
| `cf4ar_etch.yaml` | ion-enhanced etching, and ion–neutral synergy (15× the sum of the parts) | illustrative |
| `sf6o2.yaml` | **the acceptance test**: ViennaPS's own SF₆/O₂ silicon etch, written as a reaction file | every number from `SF6O2Etching::defaultParameters()`; matches the hand-written model to 0.16 % |
| `polymer_etch.yaml` | passivation competing with the etch: two solids, one deposited while the other is removed | illustrative; the forms are standard |
| `sin_peald_dis_dose.yaml` | SiNx PE-ALD, step 1 of 2: the diiodosilane dose, integrated in time — a dose saturates rather than reaching a steady state | Zeghouane et al. 2024; chemistry transcribed from Ovanesyan et al. 2015 and Ande et al. 2015 |
| `sin_peald_n2h2_plasma.yaml` | step 2 of 2: the N₂-H₂ plasma strips the iodine and grows the Si-N network; the coverages the dose left are its initial condition | same sources |

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

`demoGaAsMechanism.py` asks what a reduced mechanism costs. At the conditions
of Mountziaris and Jensen the three-reaction reduction is within 0.007 % of
all 26, and holds
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

## The eight demonstration cases

Each of the eight mechanisms below is run in a geometry suited to the process
it describes. These are the runs reported in the accompanying article, and
each writes `<stem>_initial.vtp`, `<stem>_step*.vtp`,
`<stem>_final.vtp` and `<stem>_final_volume.vtu`.

```bash
S=./surfaceChemistry
$S -r reactions/silane_inputs.mechanism.json --width 40 --depth 180 \
   --grid 1.0 --thickness 22 --film PolySi  --snapshots 4 --rays 1000 --out case1
$S -r reactions/sige_stack.mechanism.json --stack 4 --layer 12 --oxide 15 \
   --mask 10 --width 45 --depth 10 --taper 3 --grid 1.0 --thickness 8 \
   --film PolySi --snapshots 3 --rays 800 --out case2
$S -r reactions/diamond.mechanism.json --fin --width 60 --depth 90 \
   --grid 1.0 --thickness 25 --film Diamond --snapshots 4 --rays 1000 --out case3
$S -r reactions/gaas_cvd.mechanism.json      --width 80 --depth 120 \
   --grid 1.5 --thickness 20 --film GaAs    --snapshots 4 --rays 4000 --out case4
$S -r reactions/ar_sputter.mechanism.json    --width 60 --depth 0 --mask 45 \
   --grid 1.0 --thickness 55 --snapshots 4 --rays 1000 --out case5
$S -r reactions/sf6o2.mechanism.json         --width 60 --depth 0 --mask 50 \
   --grid 1.0 --thickness 100 --snapshots 4 --rays 1000 --out case6
$S -r reactions/polymer_etch.mechanism.json  --width 60 --depth 0 --mask 45 \
   --grid 1.0 --thickness 55 --snapshots 4 --rays 1500 --out case7
```

Case 1 is also run with `--sticking SiH4=100` and `--sticking SiH4=1000`,
which multiplies the adsorption rate constant of the file. The factor reaches
the rate law and the ray termination together, since the two read the same
number.

Case 6 is additionally run in the cylindrical hole that Bobinac et al.,
Micromachines 14 (2023) 665, use for this chemistry, once from the reaction
file and once from ViennaPS's own hand-written `SF6O2Etching` class, and then
over the four feed gas compositions of their Table 1. `--handwritten` swaps
the model and changes nothing else, so the two runs share a geometry, a ray
count and a process time.

```bash
G3="--gpu --width 400 --depth 0 --mask 1200 --extent 1400 --grid 20 --rays 300 --time 160"
for y in "050 7142.857 300" "044 7857.143 200" "056 5714.286 1000" "062 4285.714 1500"; do
  set -- $y
  $S -D 3 -r reactions/sf6o2.mechanism.json --flux F=$2 --flux O=$3 \
     --flux ion=10 --taper 0 $G3 --out h_y$1
done
$S -D 3 -r reactions/sf6o2.mechanism.json --flux F=7142.857 --flux O=300 \
   --flux ion=10 --handwritten --taper 0 $G3 --out h_hand
```

The fluorine fluxes are the ones that table quotes divided by the sticking,
because the hand-written model states a sticking-weighted flux where this file
states the flux and the sticking separately. `--handwritten` applies the same
conversion the other way, so the two runs describe one physical condition.

`--flux` and `--sticking` override what the file declares, and `--time` runs
for a stated time instead of removing a stated thickness, which is what
comparing gas compositions needs since the blanket rate is one of the things
that changes.

Three of the cases are run a second time with one number varied, which is
what those figures put side by side. A reverse constant is a reaction
like any other once the file is compiled, so `--rate` reaches it by the index
the run prints:

```bash
$S -r reactions/diamond.mechanism.json --rate 1=0  --fin --width 60 --depth 90 \
   --grid 1.0 --film Diamond --time 33.61 --rays 1000 --out dia_off
$S -r reactions/ar_sputter.mechanism.json --ion-energy 200 --width 60 --depth 0 \
   --mask 45 --grid 1.0 --time 50 --rays 1000 --out spu_200
$S -r reactions/polymer_etch.mechanism.json --flux CF2=500 --width 60 --depth 0 \
   --mask 45 --grid 1.0 --time 9.22 --rays 1500 --out pol_500
```

Each pair is run for the same time rather than to the same thickness, since
the rate is one of the things the varied number changes.

The eighth is the atomic layer process, run in a lateral high-aspect-ratio
cavity of the kind conformality is measured in. `--lateral` switches the
geometry from a trench to that cavity and puts the driver in micrometres, and
the run also writes `<stem>_profile.txt`, the film thickness along the cavity
on a cut at half the gap height:

```bash
for dose in 0.05 0.2 1.0; do
  ./cyclicProcess --lateral --gap-length 100 --grid 0.1 \
     --dose-file       reactions/al2o3_tma.mechanism.json \
     --coreactant-file reactions/al2o3_h2o.mechanism.json \
     --film Al2O3 --cycles 20 --dose $dose --purge 0.5 --coreactant $dose \
     --dose-steps 12 --coreactant-steps 12 --rays 500 --engine cpu \
     --max-change 2e-4 --initial O=1.0 --initial Al=1.0 \
     --out al2o3_ph_$dose
done
```

Twenty cycles saturate at 22.64 A near the opening in every one of the three,
which is twenty times the 1.132 A per cycle the same mechanism gives on the
open field, and the half-thickness penetration depth moves from 11.9 to 12.8
to 13.1 um as the dose grows. The film reaches zero at 13.5 um, so the
remaining 86 um of the cavity stays bare at the unity sticking the file
declares.

## Timing

What is worth measuring is the cost of deriving the coverage balance from a
reaction file and solving it by Newton's method at every surface point,
against evaluating a closed form derived once by hand. The figure that matters
is the share of a run the solve accounts for, since a feature-scale step
spends most of its time tracing rays, and it is the robust quantity because
its numerator and denominator come from the same run.

```bash
# pin the machine first, on a quiet system
sudo cpupower frequency-set -g performance     # if available
export OMP_NUM_THREADS=<cores>

cd <build>/examples/surfaceChemistry
python3 benchmark.py --repeats 7
```

Four rows of one geometry, a trench 80 nm wide and 120 nm deep on a 1.5 nm
grid at 2000 rays per surface point, advanced by 5 nm, so the rows differ only
in the mechanism: `silane_inputs` (3 reactions over 2 coverages), `sf6o2` (7
over 2), `sf6o2` again through `--handwritten` so the same chemistry is solved
from the closed form of the ViennaPS class, and `gaas_cvd` (30 over 7). For
each it runs `--profile`, which times the coverage solve against the wall time
of the run containing it, and `--bench 200000`, which times the solve alone
with the transport left out. Both flux engines are measured where a device is
present. Expect the device shares to be several times the host shares, since
the transport gets faster and the solve does not.

Wall times scatter, by around 40 % run to run on a laptop, which is why the
script reports medians. It needs only Python 3 from the standard library.

To check a single figure by hand:

```bash
./surfaceChemistry -r reactions/sf6o2.mechanism.json --profile \
    --width 80 --depth 120 --grid 1.5 --thickness 5 --rays 2000 --out /tmp/p
./surfaceChemistry -r reactions/sf6o2.mechanism.json --bench 200000
./surfaceChemistry -r reactions/sf6o2.mechanism.json --handwritten --bench 200000
```

## Validation

### Reference test run

```bash
python3 validation/testsuite.py            # check against the stored reference
python3 validation/testsuite.py --update   # rewrite the reference
```

This is the comprehensive test run for the package. It solves every
non-cyclic reaction file of `reactions/` at a single surface point, under
unobstructed fluxes and one ray, and checks each coverage and the resulting
surface velocity against `validation/reference.txt`. The point solve carries
no Monte Carlo sampling, so it is deterministic to the digits the driver
prints and a mismatch is a real change in the model rather than noise. It
exercises the whole chain: the mechanism data is read, the free-site
exponents and mass-action rate laws are rebuilt from the reactions, the
coverage balance is solved by the damped Newton iteration, and the velocity
is formed from the reactions that move a solid.

A cyclic mechanism has no steady state by construction, since half its
reactions have no reactant flowing at any one time, so those files are driven
instead through `cyclicProcess` and checked on their growth per cycle.

The two are held to different tolerances, for a measured reason. The
steady-state solve is a Newton root and repeats bit for bit on one build, so
it is checked to 1e-5, which is the last digit the driver prints. The growth
per cycle is integrated over many adaptive sub-steps whose order is not fixed,
and repeated runs of the same binary spread by about 1.6e-5, so it is checked
to 1e-3. A failure at that tolerance is a real change, not run-to-run noise.

Expected output, which takes well under a minute:

```
38 values checked over 12 steady-state mechanisms and the cyclic case, 0 failures
```

The rates in `reference.txt` are the "single point" column of the table of
eight mechanisms in the accompanying article, and the cyclic value is its
2.48 A/cycle.

### Diamond radical fraction

```bash
python validation/diamondRadicalFraction.py
```

Solves the diamond mechanism over 900–1400 K and compares the radical fraction
against the published closed form (0.26 % worst case). It runs no simulation: it
is a check of the framework rather than a demonstration of it.

## Running a mechanism, and writing a new one

**Running any of the nineteen mechanisms needs this directory and a ViennaPS
install.** That holds for the C++ driver, the Python driver and both demos.

**Writing a twentieth needs one install.** Compiling a reaction file -- parsing
the equations, checking the atom balance, inferring the free sites, deriving the
stoichiometry -- is ViennaChem's job:

```bash
pip install git+https://github.com/ViennaTools/ViennaChem@main
```

Then the `.yaml` path works directly. With ViennaChem absent the driver runs
every mechanism from its compiled form and announces that it has done so, and it
refuses outright where the `.yaml` is newer than the data beside it:

```
note: ViennaChem is not installed, so 'silane.mechanism.json' is used
      instead of the reaction file itself.
'silane.yaml' is NEWER than 'silane.mechanism.json': the mechanism data is
      stale. Recompile it with `python -m viennachem silane.yaml ...`.
```

The model itself is `include/viennaps/models/psSurfaceChemistry.hpp`, its file
reader `psChemicalMechanismIO.hpp`, and the device shader
`gpu/models/SurfaceChemistry.cuh`.
