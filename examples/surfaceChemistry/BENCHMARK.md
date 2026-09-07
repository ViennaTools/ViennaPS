# Timing the coverage solve

What this measures is the cost of deriving the coverage balance from a
reaction file and solving it by Newton's method at every surface point,
against evaluating a closed form derived once by hand. The figure that matters
is the **share of a run** the solve accounts for, since a feature-scale step
spends most of its time tracing rays.

Everything below assumes a quiet machine. The laptop this was drafted on gave
40 % run-to-run scatter in wall time, which is why the script reports medians
and why the share, whose numerator and denominator come from the same run, is
the robust quantity.

## Step by step, from nothing

    # 1. get the sources
    git clone https://github.com/ViennaTools/ViennaPS.git
    cd ViennaPS

    # 2. configure a Release build; add the GPU flag only if OptiX is present
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DVIENNAPS_BUILD_EXAMPLES=ON
    # or, with a device engine:
    # cmake -B build -DCMAKE_BUILD_TYPE=Release -DVIENNAPS_BUILD_EXAMPLES=ON \
    #       -DVIENNAPS_USE_GPU=ON

    # 3. build just what is needed
    cmake --build build --target surfaceChemistry -j

    # 4. pin the machine
    sudo cpupower frequency-set -g performance     # if available
    export OMP_NUM_THREADS=<cores>                 # record what you set

    # 5. run
    cd build/examples/surfaceChemistry
    python3 benchmark.py --repeats 7

That prints a summary and writes three files:

| file | what it is |
| --- | --- |
| `benchmark_results.txt` | every figure, with the machine it was taken on |
| `benchmark_table.tex`   | Table `tab:cost`, host and device columns, ready to drop in |
| `benchmark_numbers.tex` | the figures the prose quotes, as LaTeX macros |

## Putting it in the paper

Copy the two `.tex` files over the placeholders:

    cp benchmark_table.tex benchmark_numbers.tex <paper directory>/

The manuscript already carries `\input{benchmark_numbers}` in its preamble and
`\input{benchmark_table}` in Section 7.5, so no editing is needed. The
placeholders currently in the paper print `??` for every quoted figure and a
table of dashes captioned NOT YET MEASURED, so an unpopulated build is obvious
rather than plausible.

Requires only Python 3 from the standard library. ViennaChem is not needed,
since each mechanism ships compiled alongside its reaction file.

## What the script does

Four rows of one geometry, a trench 80 nm wide and 120 nm deep on a 1.5 nm
grid at 2000 rays per surface point, advanced by 5 nm. Holding the geometry
fixed means the rows differ only in the mechanism.

- `silane_inputs`, 3 reactions over 2 coverages, the smallest
- `sf6o2`, 7 reactions over 2 coverages
- `sf6o2` again through `--handwritten`, which swaps in ViennaPS's own
  `SF6O2Etching` class so the same chemistry is solved from its closed form
- `gaas_cvd`, 30 reactions over 7 coverages, the largest

For each it runs `--profile`, which wraps the surface model in one that
forwards every call and times the coverage solve, and reports that against the
wall time of the same run. It then runs `--bench 200000`, which times the
solve alone over 200000 points with the transport left out, for the reaction
file and for the closed form.

## The device engine

Both engines are measured. `--profile` resolves the device model itself and
places the same wrapper on it, since only the transport moves to the device
and the coverage solve is the same host-side class either way.

The device column is the one that tests the claim hardest. The transport gets
faster and the solve does not, so the same absolute cost becomes a larger
share of a shorter run. Expect the device shares to be several times the host
shares. If the build has no device engine the script says so and reports the
host column alone.

## If you want to check one thing by hand

    ./surfaceChemistry -r reactions/sf6o2.mechanism.json --profile \
        --width 80 --depth 120 --grid 1.5 --thickness 5 --rays 2000 --out /tmp/p

    ./surfaceChemistry -r reactions/sf6o2.mechanism.json --bench 200000
    ./surfaceChemistry -r reactions/sf6o2.mechanism.json --handwritten --bench 200000
