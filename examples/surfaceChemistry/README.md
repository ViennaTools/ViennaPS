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

CMake copies `reactions/` into the build directory when it configures, so a
driver run from there reads that copy. Edit a mechanism in the source tree and
the change reaches a run only after the copy is refreshed, either by
re-running CMake or by copying the two files across by hand.

ViennaPS reads the compiled form in C++ (`psChemicalMechanismIO.hpp`), and that
is the *only* reader: the Python driver hands the same data to the same reader
through `ps.ChemicalMechanism.fromJSON`, so there is one implementation of the
format, one shared by both languages. Every file carries a `schemaVersion`, and
a reader refuses any version it recognises as later than its own.

## The capability each reaction file demonstrates

| file | what it shows | where the numbers come from |
|---|---|---|
| `silane.yaml` | the base case: LPCVD polysilicon from silane | checked against a by-hand implementation of the same mechanism |
| `silane_inputs.yaml` | both alternative input forms at once: the SiH₄ supply as a partial pressure (mTorr), `Γ = p/√(2πmk_BT)`, and the adsorption as a rate constant in cm/s, `s = 4k_ads/v̄` | same chemistry, same answer to 0.003 % |
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
of Mountziaris and Jensen the three-step reduction is within 2e-4 % of
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
it describes. These are the runs reported in the accompanying article. Each
writes `<stem>_initial.vtp`, `<stem>_step*.vtp`, `<stem>_final.vtp` and
`<stem>_final_volume.vtu`. `GPU` is `--gpu` where the device engine is built
and empty otherwise.

```bash
S=./surfaceChemistry
GPU=""; $S -r reactions/sf6o2.mechanism.json --gpu --bench 1 >/dev/null 2>&1 && GPU="--gpu"

# 1  silane, at the file's k_ads and at 100 and 1000 times it
for f in 1 100 1000; do
  A=""; [ $f -ne 1 ] && A="--sticking SiH4=$f"
  O=case1_silane; [ $f -ne 1 ] && O=case1_silane_s$f
  $S -r reactions/silane_inputs.mechanism.json $A --width 40 --depth 180 \
     --grid 1.0 --thickness 22 --film PolySi --snapshots 4 --rays 1000 --out $O
done
# 2  selective epitaxy on a Si/SiGe superlattice
$S -r reactions/sige_stack.mechanism.json --stack 4 --layer 12 --oxide 15 \
   --mask 10 --width 45 --depth 10 --taper 3 --grid 1.0 --thickness 8 \
   --film PolySi --snapshots 3 --rays 800 --out case2_sige
# 3  diamond over a fin, reverse constant as stated, off, and x10
D="--fin --width 60 --depth 90 --grid 1.0 --film Diamond --snapshots 3 \
   --rays 1000 --time 33.61 --height 260 $GPU"
$S -r reactions/diamond.mechanism.json             $D --out dia_pub
$S -r reactions/diamond.mechanism.json --rate 1=0  $D --out dia_off
$S -r reactions/diamond.mechanism.json --rate 1=1  $D --out dia_x10
# 4  GaAs MOVPE in a trench
$S -r reactions/gaas_cvd.mechanism.json --width 80 --depth 120 --grid 1.5 \
   --thickness 20 --film GaAs --snapshots 4 --rays 4000 --out case4_gaas
# 5  argon sputtering at two mean ion energies
A="--width 60 --depth 0 --mask 45 --grid 1.0 --snapshots 3 --rays 1000 \
   --time 50 --height 300 $GPU"
$S -r reactions/ar_sputter.mechanism.json --ion-energy 100 $A --out spu_100
$S -r reactions/ar_sputter.mechanism.json --ion-energy 200 $A --out spu_200
# 7  the fluorocarbon etch at two precursor fluxes, the polymer given a
#    material of its own so the film is distinguishable from the substrate
P="--width 60 --depth 0 --mask 45 --grid 1.0 --snapshots 8 --rays 1500 \
   --time 10 --height 220 --substrate SiO2 --film Polymer $GPU"
$S -r reactions/polymer_etch.mechanism.json                $P --out pol_5
$S -r reactions/polymer_etch.mechanism.json --flux CF2=100 $P --out pol_100
```

`--sticking SiH4=f` multiplies the prefactor of the adsorption steps that
consume the species. The ray tracer absorbs a particle with the rate laws of
the reactions that consume it, so the factor reaches the surface solve and the
transport together. `--rate i=v` sets the prefactor of reaction `i`, by the
index the run prints. A reverse constant is a reaction like any other once the
file is compiled. `--flux` overrides the flux a file declares, `--ion-energy`
the mean ion energy, and `--time` runs for a stated time instead of growing or
removing a stated thickness, which is what comparing two variants of one case
needs since the rate is one of the things that changes.

Case 6 is run in the cylindrical hole that Bobinac et al., Micromachines 14
(2023) 665, use for this chemistry, over the four feed gas compositions of
their Table 1, and once more from ViennaPS's own hand-written `SF6O2Etching`
class. `--handwritten` swaps the model and changes nothing else, so the two
runs share a geometry, a ray count and a process time.

```bash
G3="--width 400 --depth 0 --mask 1200 --extent 1400 --grid 20 --rays 300 \
    --time 160 $GPU"
for y in "050 5000 300" "044 5500 200" "056 4000 1000" "062 3000 1500"; do
  set -- $y
  $S -D 3 -r reactions/sf6o2.mechanism.json --flux F=$2 --flux O=$3 \
     --flux ion=10 --taper 0 $G3 --out h_y$1
done
$S -D 3 -r reactions/sf6o2.mechanism.json --flux F=5000 --flux O=300 \
   --flux ion=10 --handwritten --taper 0 $G3 --out h_hand
```

The fluxes are those of that table as printed. They are incident fluxes, to
which their Eq. (4) applies the sticking of 0.7 separately, as this file does.
The hand-written model takes a sticking-weighted flux instead, and
`--handwritten` converts the file's flux accordingly, so the two runs describe
one physical condition.

The eighth is the atomic layer process, run in a lateral high-aspect-ratio
cavity of the kind conformality is measured in. `--lateral` switches the
geometry from a trench to that cavity and puts the driver in micrometres, and
the run also writes `<stem>_profile.txt`, the film thickness along the cavity
on a cut at half the gap height. The article varies the TMA dose with a water
exposure of 10 s, four times what the water half-cycle needs to saturate, and
runs the cavity on the GPU line engine. A molecule collides with the walls
thousands of times in this cavity, and the CPU disk engine discards about 3e-5
of the rays per collision through its rule for back-face hits, which in a cavity
that absorbs nothing leaves 0.3 of the source flux at the closed end. The line
engine needs ViennaRay's line intersection to let neighbouring segments overlap
(gpu/pipelines/GeneralPipelineLine.cu accepts hits up to 1e-5 of the segment
length beyond either end). With a constant sticking of 1e-3, the flux of the
line engine along the cavity agrees with an independent free-molecular Monte
Carlo calculation of the same geometry to within 2.2 % in every 10 um of depth
(papers/reproduce/cavity_transport_check.py in the article repository):

```bash
for dose in 0.05 0.1 0.14 0.2 0.28 0.4 0.8; do
  ./cyclicProcess --lateral --gap-length 100 --grid 0.1 \
     --dose-file       reactions/al2o3_tma.mechanism.json \
     --coreactant-file reactions/al2o3_h2o.mechanism.json \
     --film Al2O3 --cycles 20 --dose $dose --purge 0.5 --coreactant 10 \
     --dose-steps 12 --coreactant-steps 12 --rays 500 --engine gpu \
     --max-change 2e-4 --initial O=1.0 --initial Al=1.0 \
     --out al2o3_ph_$dose
done
```

The profile starts at the symmetry plane of the 10 um opening, so the cavity
runs from x = 5 um (its entrance) to x = 105 um, where the end wall is read as
film. For doses of 0.14 s and longer the film saturates at the entrance at
22.6 A, twenty times the 1.131 A per cycle the same mechanism gives on the open
field; at 0.1 and 0.05 s it reaches 99 and 89 % of that. Measured from the
entrance, the half-thickness penetration depth is 31.2, 58.5 and 85.2 um at
0.05, 0.1 and 0.14 s, and from 0.2 s on the film is above half thickness up to
the closed end. TMA reacts with a probability of 1e-3 per collision with a
hydroxylated wall, so the front is broad. Close to the closed end the film rises
again by about 1 % of the saturated thickness, because the end wall re-emits
diffusely the molecules that arrive nearly parallel to the cavity walls. The growth per cycle
the driver prints is the largest over all surface points; the open-field value
is the mean film on the top surface of the structure.

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
in the mechanism: `silane_inputs` (3 reactions over 2 coverages), `sf6o2` (9
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
