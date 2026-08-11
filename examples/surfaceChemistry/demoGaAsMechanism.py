#!/usr/bin/env python3
"""The full GaAs MOVPE mechanism against its three-reaction reduction.

`gaas_full.yaml` is all 26 surface reactions of Mountziaris & Jensen,
J. Electrochem. Soc. 138, 2426 (1991), Table II, for the (110) surface.
`gaas_reversible.yaml` is the same mechanism cut to the three reactions their
sensitivity analysis calls dominant: [S5], [S11] and [S22].

Reducing a mechanism is a judgement call, and the judgement is only good over
some range of conditions. This runs both on the same gas phase and finds where
they part company:

  1. at the paper's conditions, how close are they?
  2. across temperature, where does the reduction start to drift?
  3. the reduced model drops the methyl chemistry entirely. Feed the surface
     methyl radicals and watch a second growth channel open.

Everything here is analytic -- coverages and a blanket rate -- so it needs only
ViennaChem. The last section runs both in a trench, which needs ViennaPS.

    python demoGaAsMechanism.py [--trench]
"""

import argparse
import copy

import viennachem as vc

FULL = "gaas_full.yaml"
REDUCED = "gaas_reversible.yaml"


def solve(data, temperature=None):
    """Coverages and growth rate, as a dict keyed by coverage name."""
    theta = vc.solve_coverages(data, temperature=temperature)
    v = vc.growth_rate(data, theta, temperature=temperature)
    names = [c["name"] for c in data["coverages"]]
    return dict(zip(names, theta)), v


def with_flux(data, species, flux):
    """A copy of the mechanism with one incident flux changed."""
    out = copy.deepcopy(data)
    for g in out["gas"]:
        if g["name"] == species:
            g["flux"] = flux
            return out
    raise KeyError(f"{species} is not a gas species of {out['name']}")


def reaction_rates(data, theta, temperature=None):
    """Every reaction's rate at a given steady state."""
    ks = vc.rate_constants(data, temperature)
    gammas = vc.default_fluxes(data)
    frees = vc.free_fractions(data, theta)
    return [(rx["id"], rx["eq"], vc.rate(rx, ks[j], gammas, theta, frees))
            for j, rx in enumerate(data["reactions"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trench", action="store_true",
                    help="also grow both mechanisms in a trench (needs ViennaPS)")
    ap.add_argument("--gpu", action="store_true",
                    help="trace the trench on the device")
    args = ap.parse_args()

    full = vc.from_file(FULL)
    reduced = vc.from_file(REDUCED)

    print("GaAs MOVPE: 26 reactions against 3\n")
    for label, d in (("full", full), ("reduced", reduced)):
        gas = [g["name"] for g in d["gas"] if g["flux"]]
        print(f"  {label:8s} {len(d['reactions']):2d} reactions   "
              f"{len(d['coverages'])} coverages   "
              f"{len(d['solids'])} solid(s)   supplied: {gas}")

    # --- 1. the paper's conditions -----------------------------------------
    print(f"\n1) at {full['constants']['temperature']:.0f} K, on the same gas phase")
    th_f, v_f = solve(full)
    th_r, v_r = solve(reduced)
    for name in th_r:
        print(f"   theta_{name:8s}  full {th_f[name]:.6e}   reduced {th_r[name]:.6e}")
    print(f"   growth       full {v_f:.4f}   reduced {v_r:.4f} nm/s"
          f"   ({abs(v_f - v_r) / abs(v_r) * 100:.3f}%)")

    print("\n   every reaction carrying flux, with the reduced model's own")
    print("   three marked, so the rest is what the reduction dropped:")
    rates = reaction_rates(full, [th_f[c["name"]] for c in full["coverages"]])
    kept = {"R5", "R5r", "R11", "R11r", "R23"}   # [S5], [S11], [S22] and reverses
    for rid, eq, r in rates:
        if r > 1e-6:
            mark = "  (kept)" if rid in kept else ""
            print(f"      {rid:<5} {eq:<40} {r:12.4e}{mark}")

    # --- 2. across temperature ---------------------------------------------
    print("\n2) across temperature: does the reduction hold?")
    print("     T [K]     full        reduced     difference   theta_AsH (full)")
    for T in range(800, 1301, 100):
        th_f, v_f = solve(full, T)
        th_r, v_r = solve(reduced, T)
        rel = (v_f - v_r) / v_r * 100 if v_r else float("nan")
        print(f"    {T:6d}  {v_f:10.4f}  {v_r:10.4f}   {rel:+8.4f}%   "
              f"{th_f['AsH*']:.4f}")
    print("   Both peak near 1000 K, for the reason the reduced model already")
    print("   captures: desorption of the two adsorbates overtakes growth.")

    # --- 3. the chemistry the reduction leaves out --------------------------
    print("\n3) the methyl chemistry, which the reduction has no room for.")
    print("   TMG pyrolysis releases CH3; the reduced model cannot see it.")
    print("     Gamma_CH3     full      reduced   difference   theta_As*   "
          "growth via [S26]")
    base = next(g["flux"] for g in full["gas"] if g["name"] == "GaCH3")
    _, v_r = solve(reduced)
    for frac in (0.0, 0.01, 0.1, 0.5, 1.0, 3.0):
        f = with_flux(full, "CH3", frac * base)
        th_f, v_f = solve(f)
        rates = reaction_rates(f, [th_f[c["name"]] for c in f["coverages"]])
        via26 = next(r for rid, _, r in rates if rid == "R26")
        via23 = next(r for rid, _, r in rates if rid == "R23")
        rel = (v_f - v_r) / v_r * 100
        print(f"    {frac * base:10.3e} {v_f:9.4f} {v_r:9.4f}  {rel:+8.3f}%   "
              f"{th_f['As*']:.3e}     {via26 / (via23 + via26) * 100:5.2f}%")

    print("\n   A methyl radical strips the hydrogen off an adsorbed AsH,")
    print("   [S24], leaving a bare arsenic. Growth on that arsenic, [S26],")
    print("   crosses a 20 kcal/mol barrier against the 29.3 of the main step,")
    print("   [S22], so it is the faster channel wherever it can run. The")
    print("   reduction, having neither As* nor a methyl coverage, holds its")
    print("   rate whatever the gas phase does.")

    if args.trench:
        trench(full, reduced, args.gpu)


def trench(full, reduced, gpu=False):
    """Both mechanisms through the same feature."""
    import json
    import numpy as np
    import viennaps as ps
    import viennals
    import viennals.d2 as ls2

    ps.setDimension(2)
    ps.setNumThreads(16)
    ps.Length.setUnit("nm")
    ps.Time.setUnit("s")
    ps.Logger.setLogLevel(ps.LogLevel.WARNING)

    print("\n4) the same trench, grown by each")
    for label, data in (("full", full), ("reduced", reduced)):
        mech = ps.ChemicalMechanism.fromJSON(json.dumps(data))
        dom = ps.Domain(gridDelta=2.0, xExtent=200.0, yExtent=200.0)
        ps.MakeTrench(domain=dom, trenchWidth=80.0, trenchDepth=120.0,
                      material=ps.Material.Si).apply()
        dom.duplicateTopLevelSet(ps.Material.GaAs)

        def bounds():
            m = viennals.Mesh()
            ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
            n = np.array(m.getNodes())
            return n[:, 1].max(), n[:, 1].min()

        t0, b0 = bounds()
        p = ps.Process(dom, ps.SurfaceChemistry(mech), 1.0)
        p.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE if gpu
                            else ps.FluxEngineType.CPU_DISK)
        c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 20
        p.setParameters(c)
        r = ps.RayTracingParameters(); r.raysPerPoint = 1000; p.setParameters(r)
        p.apply()
        t1, b1 = bounds()
        field, bottom = t1 - t0, b1 - b0
        print(f"   {label:8s} field {field:7.3f} nm   bottom {bottom:7.3f} nm"
              f"   step coverage {bottom / field:.4f}")
        dom.saveSurfaceMesh(filename=f"gaas_{label}_final.vtp", addInterfaces=True)


if __name__ == "__main__":
    main()
