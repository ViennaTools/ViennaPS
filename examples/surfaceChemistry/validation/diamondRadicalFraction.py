#!/usr/bin/env python3
"""The diamond mechanism against its published closed form.

The "standard growth model" for CVD diamond (Bristol CVD Diamond Group,
https://www.chm.bris.ac.uk/pt/diamond/growthmodel.htm) with the rate parameters
of the SURFACE CHEMKIN mechanism in the accompanying thesis
(https://www.chm.bris.ac.uk/pt/diamond/rolythesis/chapter8.htm).

A diamond surface carbon is either terminated by hydrogen or left as a radical.
Growth happens ONLY on radical sites, and the radical fraction is set by a
REVERSIBLE hydrogen abstraction. In `../reactions/diamond.yaml` the free site "*"
IS the radical, so the radical fraction is R = theta_* = 1 - theta_H.

Solving the coverage balance reproduces the published closed form

    R = 1 / (1 + 0.3 exp(3430/Ts) + 0.1 exp(-4420/Ts) [H2]/[H])

whose last term exists ONLY because the abstraction runs backwards. This is a
check rather than a demonstration, which is why it is not part of the driver.

    python validateDiamond.py
"""

import math
import os
import sys

import viennaps as ps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "reactions", "diamond.mechanism.json")

# hot-filament conditions: 20 torr, 1% CH4 in H2
TORR = 133.322
P_TOTAL = 20.0 * TORR
X = {"H": 5.0e-3, "H2": 0.98, "CH3": 5.0e-5}
M = {"H": 1.008e-3, "H2": 2.016e-3, "CH3": 15.035e-3}  # kg/mol
N_A, KB = 6.02214076e23, 1.380649e-23


def hertz_knudsen(species, temperature):
    """The kinetic-theory wall flux, in 1e15 /cm2 /s."""
    p = X[species] * P_TOTAL
    m = M[species] / N_A
    return p / math.sqrt(2.0 * math.pi * m * KB * temperature) * 1e-4 / 1e15


def published(Ts, h2_over_h):
    return 1.0 / (1.0 + 0.3 * math.exp(3430.0 / Ts)
                  + 0.1 * math.exp(-4420.0 / Ts) * h2_over_h)


def main():
    ps.setDimension(2)
    ps.Length.setUnit("nm")
    ps.Time.setUnit("s")
    ps.Logger.setLogLevel(ps.LogLevel.WARNING)

    if not os.path.exists(DATA):
        sys.exit(f"no mechanism data at {DATA}")

    print("Diamond CVD, standard growth model (Bristol CVD Diamond Group)")
    print("  20 torr, 1% CH4 in H2\n")
    print(" Ts [K]     R solved   R published   rel. diff   growth [um/h]")
    print("-" * 62)

    worst = 0.0
    for Ts in range(900, 1401, 100):
        mech = ps.ChemicalMechanism.fromFile(DATA)
        mech.temperature = float(Ts)
        # the fluxes follow the temperature, so they are recomputed per point
        gamma = [hertz_knudsen(s, Ts) for s in ("H", "H2", "CH3")]
        theta = mech.solveCoverages(gamma, [0.0] * len(mech.coverageNames))
        R = 1.0 - theta[0]
        R_pub = published(Ts, gamma[1] / gamma[0])
        rate = mech.growthRate(gamma, theta) * 3600.0 / 1000.0
        rel = abs(R - R_pub) / R_pub
        worst = max(worst, rel)
        print(f"{Ts:7d}   {R:.5e}   {R_pub:.5e}   {rel:.2e}   {rate:12.3f}")

    print("-" * 62)
    print(f"  worst relative difference {worst:.2e}")
    print("  The small residual is the methyl growth term, which the published")
    print("  closed form neglects.")


if __name__ == "__main__":
    main()
