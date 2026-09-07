#!/usr/bin/env python3
"""Reference test run for the SurfaceChemistry model.

Solves every reaction file of this directory at a single surface point, under
unobstructed fluxes and a single ray, and checks the result against the
reference output stored beside this script. The point solve carries no Monte
Carlo sampling, so it is deterministic and a mismatch is a real change in the
model rather than noise.

What is exercised for each file is the whole chain: the
mechanism data is read, the free-site exponents and mass-action rate laws are
rebuilt from the reactions, the coverage balance is solved by the damped
Newton iteration, and the surface velocity is formed from the reactions that
move a solid.

The steady-state numbers are reproducible bit for bit on one build. The
cyclic one is not, since it integrates over sub-steps whose order is not
fixed, so it is checked to a looser tolerance.

    python3 testsuite.py                 check against reference.txt
    python3 testsuite.py --update        rewrite reference.txt
    python3 testsuite.py --driver PATH   use a driver other than the default

The driver is looked for in ../../../build/examples/surfaceChemistry, which is
where an in-tree build puts it, or given with --driver.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXDIR = os.path.dirname(HERE)
REFERENCE = os.path.join(HERE, 'reference.txt')
DEFAULT_DRIVER = os.path.normpath(os.path.join(
    EXDIR, os.pardir, os.pardir, 'build', 'examples', 'surfaceChemistry',
    'surfaceChemistry'))

# A point carries one material, so a mechanism whose rate constants are keyed
# by material is told which one to solve on.
EXTRA = {'polymer_etch': ['--substrate', 'SiO2', '--film', 'Polymer'],
         'sige_stack':   ['--substrate', 'Si'],
         'silane_selective': ['--substrate', 'Si']}

# A cyclic mechanism has no steady state by construction. Half of its
# reactions have no reactant flowing at any one time, so the point solve
# returns the trivial root and tests nothing. These are covered instead by
# solve_cycle below, which drives them through the transient path they use.
SKIP = {'al2o3_tma', 'al2o3_h2o', 'al2o3_travis_tma', 'toy_dose',
        'toy_coreactant', 'sin_peald_dis_dose', 'sin_peald_n2h2_plasma'}

POINT = ['--width', '0', '--depth', '0', '--mask', '0', '--grid', '100',
         '--thickness', '1', '--rays', '1']

# Relative tolerances, set from measured reproducibility rather than guessed.
# The steady-state solve is a Newton root and repeats bit for bit on one
# build, so the only slack it needs is the last printed digit. The cyclic
# value is integrated over many adaptive sub-steps whose order is not fixed,
# and repeated runs of the same binary spread by about 1.6e-5, so it is given
# room well above that and above what a different compiler would add.
TOL = 1e-5
TOL_CYCLE = 1e-3


def cases():
    d = os.path.join(EXDIR, 'reactions')
    return sorted(f[:-len('.mechanism.json')] for f in os.listdir(d)
                  if f.endswith('.mechanism.json') and f[:-len('.mechanism.json')] not in SKIP)


def solve(driver, mech, rundir):
    p = subprocess.run(
        [driver, '-r', 'reactions/%s.mechanism.json' % mech] + POINT
        + ['--out', os.path.join(rundir, 'testsuite_tmp')] + EXTRA.get(mech, []),
        capture_output=True, text=True, cwd=EXDIR)
    out = p.stdout + p.stderr
    vals = {}
    for m in re.finditer(r'theta_(\S+)\s*=\s*(-?[0-9.eE+-]+)', out):
        vals['theta_' + m.group(1)] = float(m.group(2))
    # the driver already signs the rate, negative where the surface recedes
    m = re.search(r'(?:growth|etch) rate\s*=\s*(-?[0-9.eE+-]+)', out)
    if m:
        vals['rate_nm_per_s'] = float(m.group(1))
    return vals, p.returncode, out


def solve_cycle(driver, rundir):
    """The transient path, which no steady-state solve reaches.

    Runs the two-mechanism toy cycle on a flat surface at
    one ray, so the growth per cycle is set by the chemistry and the
    integrator alone."""
    cyc = os.path.join(os.path.dirname(driver), 'cyclicProcess')
    if not os.path.exists(cyc):
        return {}, 0, 'cyclicProcess not built'
    p = subprocess.run(
        [cyc, '--width', '0', '--depth', '0', '--grid', '2.0',
         '--dose-file', 'reactions/toy_dose.mechanism.json',
         '--coreactant-file', 'reactions/toy_coreactant.mechanism.json',
         '--cycles', '3', '--dose', '0.5', '--purge', '1.0',
         '--coreactant', '2.0', '--dose-steps', '4', '--coreactant-steps', '4',
         '--rays', '1', '--engine', 'cpu', '--max-change', '2e-4',
         '--initial', 'R*=1.0',
         '--out', os.path.join(rundir, 'testsuite_cyc')],
        capture_output=True, text=True, cwd=EXDIR)
    out = p.stdout + p.stderr
    m = re.search(r'growth per cycle[^=]*=\s*(-?[0-9.eE+-]+)', out)
    vals = {'growth_per_cycle_A': float(m.group(1))} if m else {}
    return vals, p.returncode, out


def render(mech, vals):
    return ''.join('%-22s %-24s %.9e\n' % (mech, k, v)
                   for k, v in sorted(vals.items()))


def read_reference():
    ref = {}
    if not os.path.exists(REFERENCE):
        return ref
    for line in open(REFERENCE):
        if not line.strip() or line.startswith('#'):
            continue
        mech, key, val = line.split()
        ref.setdefault(mech, {})[key] = float(val)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true',
                    help='rewrite reference.txt from this run')
    ap.add_argument('--driver', default=DEFAULT_DRIVER)
    a = ap.parse_args()
    if not os.path.exists(a.driver):
        sys.exit('driver not found at %s\nbuild ViennaPS first, or pass '
                 '--driver' % a.driver)

    rundir = os.environ.get('TMPDIR', '/tmp')
    ref = read_reference()
    text, failures, checked = [], [], 0
    for mech in cases():
        vals, code, out = solve(a.driver, mech, rundir)
        if code != 0 or not vals:
            failures.append('%-22s driver failed (exit %d)' % (mech, code))
            continue
        text.append(render(mech, vals))
        if a.update:
            continue
        if mech not in ref:
            failures.append('%-22s not in reference.txt' % mech)
            continue
        for k, v in sorted(vals.items()):
            if k not in ref[mech]:
                failures.append('%-22s %s missing from reference' % (mech, k))
                continue
            r = ref[mech][k]
            checked += 1
            if abs(v - r) > TOL * max(abs(r), 1e-30):
                failures.append('%-22s %-24s got %.9e want %.9e'
                                % (mech, k, v, r))

    vals, code, out = solve_cycle(a.driver, rundir)
    if vals:
        text.append(render('toy_cycle', vals))
        if not a.update:
            for k, v in sorted(vals.items()):
                r = ref.get('toy_cycle', {}).get(k)
                if r is None:
                    failures.append('%-22s %s missing from reference'
                                    % ('toy_cycle', k))
                    continue
                checked += 1
                if abs(v - r) > TOL_CYCLE * max(abs(r), 1e-30):
                    failures.append('%-22s %-24s got %.9e want %.9e'
                                    % ('toy_cycle', k, v, r))
    elif code != 0:
        failures.append('%-22s cycle driver failed (exit %d)' % ('toy_cycle', code))

    if a.update:
        with open(REFERENCE, 'w') as f:
            f.write('# Reference output of the SurfaceChemistry point solve.\n'
                    '# Regenerate with: python3 testsuite.py --update\n'
                    '# columns: mechanism, quantity, value\n')
            f.write(''.join(text))
        print('wrote %s' % REFERENCE)
        return 0

    for line in failures:
        print('FAIL  ' + line)
    print('\n%d values checked over %d steady-state mechanisms and the '
          'cyclic case, %d failures' % (checked, len(cases()), len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
