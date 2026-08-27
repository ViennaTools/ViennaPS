"""One PE-ALD cycle of SiNx, from the two step files, integrated in time.

    dose    sin_peald_dis_dose.yaml,   Gamma(SiH2I2) on
    purge   sin_peald_dis_dose.yaml,   nothing flowing
    plasma  sin_peald_n2h2_plasma.yaml, radicals and ion on
    purge   sin_peald_n2h2_plasma.yaml, nothing flowing

with the coverages carried from each step into the next. This mirrors
ChemicalMechanism::stepCoverages in ViennaPS -- the same production/loss split,
the same sub-step control, the same clamp -- so it is a zero-dimensional
reference for that solver, and the place to fit the rate constants against the
paper's saturation curves before any geometry is involved.

    python sin_peald_cycle.py                # cycles to the steady GPC
    python sin_peald_cycle.py --scan dose    # GPC vs DIS dose time   (Fig. 3a)
    python sin_peald_cycle.py --scan plasma  # GPC vs plasma time     (Fig. 3b)
    python sin_peald_cycle.py --scan h2      # GPC vs H radical flux  (Fig. 3c)
    python sin_peald_cycle.py --converge     # error vs maxChange

Needs ViennaChem (pip install git+https://github.com/ViennaTools/ViennaChem).
"""
import argparse, math, os, sys

sys.path.insert(0, os.environ.get("VIENNACHEM", ""))
import viennachem as vc
from viennachem.evaluate import (rate_constants, rate, free_fractions,
                                 solid_density)

HERE = os.path.dirname(os.path.abspath(__file__))
DOSE = vc.from_file(os.path.join(HERE, "sin_peald_dis_dose.yaml"))
PLASMA = vc.from_file(os.path.join(HERE, "sin_peald_n2h2_plasma.yaml"))

# The two files of one cycle must agree on the surface they describe, because
# the coverages of one step are the initial condition of the next.
COV = [c["name"] for c in DOSE["coverages"]]
if COV != [c["name"] for c in PLASMA["coverages"]]:
    sys.exit("the two step files declare different coverages:\n"
             f"  dose  : {COV}\n"
             f"  plasma: {[c['name'] for c in PLASMA['coverages']]}")

K = {id(DOSE): rate_constants(DOSE), id(PLASMA): rate_constants(PLASMA)}
TINY = sys.float_info.min


def coverage_scale(ir):
    """1e15/S0: fluxes are in 1e15 /cm2/s, so this turns a rate into ML/s."""
    S0 = ir["siteTypes"][0].get("density")
    return 1e15 / S0 if S0 else 1.0


def gammas(ir, flowing, scale=None):
    """Source fluxes for one step; `flowing` false is a purge."""
    out = []
    for g in ir["gas"]:
        v = (g["flux"] or 0.0) if (flowing and g["flux"] is not None) else 0.0
        if scale and g["name"] in scale:
            v *= scale[g["name"]]
        out.append(v)
    return out


def step_coverages(ir, gam, theta, dt, max_change=5e-4, max_sub=1000000):
    """Port of ChemicalMechanism::stepCoverages -- exponential Euler."""
    k = K[id(ir)]
    n = len(theta)
    s = coverage_scale(ir)
    elapsed = 0.0
    for _ in range(max_sub):
        if elapsed >= dt:
            break
        frees = free_fractions(ir, theta)
        P = [0.0] * n
        L = [0.0] * n
        for j, rx in enumerate(ir["reactions"]):
            rj = rate(rx, k[j], gam, theta, frees) * s
            if rj == 0.0:
                continue
            for i in range(n):
                nu = rx["nu"][i]
                if nu > 0.0:
                    P[i] += nu * rj
                elif nu < 0.0:
                    L[i] += -nu * rj / max(theta[i], TINY)

        fastest = max((abs(P[i] - L[i] * theta[i]) for i in range(n)), default=0.0)
        h = dt - elapsed
        if fastest > 0.0:
            h = min(h, max_change / fastest)

        for i in range(n):
            if L[i] * h > 1e-8:
                steady = P[i] / L[i]
                nxt = steady + (theta[i] - steady) * math.exp(-L[i] * h)
            else:
                nxt = theta[i] + h * (P[i] - L[i] * theta[i])
            theta[i] = min(1.0, max(0.0, nxt))
        total = sum(theta)
        if total > 1.0:                       # one site type here
            theta = [t / total for t in theta]
        elapsed += h
    return theta


def growth(ir, gam, theta):
    """nm/s from the solid-forming steps. Zero in the dose file, by design."""
    k = K[id(ir)]
    frees = free_fractions(ir, theta)
    return sum(a * rate(rx, k[j], gam, theta, frees) / solid_density(ir, si)
               for j, rx in enumerate(ir["reactions"])
               for si, a in enumerate(rx["solidAtoms"]) if a)


def cycle(theta, dose=0.2, purge1=3.0, plasma=15.0, purge2=3.0,
          dt=1e-2, max_change=5e-4, scale=None):
    """One dose/purge/plasma/purge cycle. Returns the coverages and the GPC."""
    grown = 0.0
    steps = ((DOSE, dose, True), (DOSE, purge1, False),
             (PLASMA, plasma, True), (PLASMA, purge2, False))
    for ir, duration, flowing in steps:
        gam = gammas(ir, flowing, scale)
        t = 0.0
        while t < duration - 1e-12:
            h = min(dt, duration - t)
            grown += growth(ir, gam, theta) * h
            theta = step_coverages(ir, gam, theta, h, max_change)
            t += h
    return theta, grown


def start():
    """An NHx-terminated surface, which is what the plasma step leaves."""
    theta = [0.0] * len(COV)
    theta[COV.index("NH2*")] = 0.85
    theta[COV.index("NH*")] = 0.05
    return theta


def settle(cycles=20, **kw):
    theta = start()
    gpc = 0.0
    for _ in range(cycles):
        theta, gpc = cycle(theta, **kw)
    return theta, gpc


def show(theta, label):
    print(f"  {label} " + " ".join(f"{n}={t:.3f}" for n, t in zip(COV, theta))
          + f"  free={1 - sum(theta):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", choices=["dose", "plasma", "h2"])
    ap.add_argument("--converge", action="store_true")
    ap.add_argument("--max-change", type=float, default=5e-4)
    a = ap.parse_args()
    mc = a.max_change

    if a.converge:
        print("Si adsorbed in one 0.2 s dose, vs the sub-step accuracy knob")
        print("(first order, so halving maxChange should halve the error)\n")
        prev = None
        for m in (2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4):
            th = step_coverages(DOSE, gammas(DOSE, True), start(), 0.2, m)
            si = th[COV.index("NHSiH2*")] + th[COV.index("NHSiH2I*")]
            d = "" if prev is None else f"   change = {abs(si - prev):.2e}"
            print(f"  maxChange = {m:<8g} Si = {si:.6f} ML{d}")
            prev = si
        return

    if a.scan is None:
        print(f"dose   : {DOSE['name']}   T = {DOSE['constants']['temperature']} K")
        print(f"plasma : {PLASMA['name']}")
        print(f"coverages carried across steps: {' '.join(COV)}\n")
        theta = start()
        show(theta, "start   ")
        for c in range(1, 11):
            theta, gpc = cycle(theta, max_change=mc)
            if c in (1, 2, 5, 10):
                show(theta, f"cycle {c:2d}")
                print(f"            GPC = {gpc * 10:.3f} A/cycle")
        print(f"\n  steady GPC = {gpc * 10:.3f} A/cycle"
              f"   (measured: 0.36-0.40 A/cycle at 300 C)")
        return

    if a.scan == "dose":
        print("DIS dose [ms]   GPC [A/cycle]     (Fig. 3a: 0.33 -> 0.40)")
        for d in (10, 25, 50, 100, 200, 300, 600):
            _, g = settle(dose=d / 1000.0, max_change=mc)
            print(f"  {d:6d}        {g * 10:.3f}")
    elif a.scan == "plasma":
        print("plasma [s]      GPC [A/cycle]     (Fig. 3b: soft saturation to ~30 s)")
        for p in (2, 5, 10, 15, 20, 30, 60):
            _, g = settle(plasma=float(p), max_change=mc)
            print(f"  {p:6d}        {g * 10:.3f}")
    elif a.scan == "h2":
        print("H flux factor   GPC [A/cycle]     (Fig. 3c: H2 gives ~3x)")
        # NH is a product of the N-H plasma chemistry, so it scales away with
        # the hydrogen; scaling H alone would leave a phantom NH flux at 0% H2.
        for f in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            _, g = settle(scale={"H": f, "NH": f}, max_change=mc)
            print(f"  {f:6.2f}        {g * 10:.3f}")


if __name__ == "__main__":
    main()
