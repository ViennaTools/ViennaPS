#!/usr/bin/env python3
"""Passivation competing with the etch, from `polymer_etch.yaml` alone.

A fluorocarbon film deposits everywhere while ions sputter it away. The trench
floor faces the ion source and loses its film, so the silicon under it etches;
the sidewalls see almost no ion flux, keep their film, and stop etching. That
is the sidewall passivation a Bosch-type process relies on.

Two solids in one mechanism: the polymer (rho 2.2) is deposited while the
silicon (rho 5.02) is removed, and each step divides by the density of the
solid IT moves.

The rate constants are illustrative; the forms are the standard ones. Run with
the ViennaPS venv:  .venv/bin/python polymer_demo.py
"""

import numpy as np
import viennaps as ps
import viennals
import viennals.d2 as ls2

from surfaceChemistry import REACTIONS, load, predict

ps.setDimension(2)
ps.setNumThreads(16)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)

MECH, _ = load(REACTIONS / "polymer_etch.yaml", None)
TIME = 4.0


def nodes(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def wall_position(pts, y_lo, y_hi):
    """Where the right-hand wall sits, averaged over a depth band."""
    band = pts[(pts[:, 1] > y_lo) & (pts[:, 1] < y_hi) & (pts[:, 0] > 0)]
    return band[:, 0].min() if len(band) else np.nan


def main():
    print("Passivation competing with the etch\n")

    print(f"mechanism : {MECH.name}\n")
    print("what the mechanism predicts per material, at normal incidence:")
    for mat in ("Si", "Polymer", "Mask"):
        th, v = predict(MECH, mat)
        what = "etches" if v < 0 else "grows "
        print(f"   {mat:8s} {what} {v:+8.4f} nm/s     "
              f"theta_CF2* {th[0]:.4f}  theta_F* {th[1]:.4f}")
    print("   the film is cleared where the ions strike, and grows where they "
          "do not\n")

    dom = ps.Domain(gridDelta=0.5, xExtent=160.0, yExtent=160.0)
    ps.MakeTrench(domain=dom, trenchWidth=50.0, trenchDepth=30.0,
                  maskHeight=25.0, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    # the film is a level set of its own, so the surface material tells the
    # mechanism whether it is looking at film or at substrate
    dom.duplicateTopLevelSet(ps.Material.Polymer)
    before = nodes(dom)
    dom.saveSurfaceMesh(filename="polymer_initial.vtp", addInterfaces=True)
    dom.saveVolumeMesh("polymer_initial")

    process = ps.Process(dom, ps.SurfaceChemistry(MECH), TIME)
    process.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    cov = ps.CoverageParameters(); cov.tolerance = 1e-4; cov.maxIterations = 20
    process.setParameters(cov)
    ray = ps.RayTracingParameters(); ray.raysPerPoint = 3000
    process.setParameters(ray)
    process.apply()

    after = nodes(dom)
    dom.saveSurfaceMesh(filename="polymer_final.vtp", addInterfaces=True)
    dom.saveVolumeMesh("polymer_final")

    floor0, floor1 = before[:, 1].min(), after[:, 1].min()
    top0, top1 = before[:, 1].max(), after[:, 1].max()
    # the sidewall band, above the original floor and below the mask
    lo, hi = floor0 + 5.0, floor0 + 20.0
    wall0, wall1 = wall_position(before, lo, hi), wall_position(after, lo, hi)

    print(f"after {TIME:g} s:")
    print(f"   trench floor   {floor0:8.2f} -> {floor1:8.2f} nm   "
          f"({floor1 - floor0:+.2f}, etched)")
    print(f"   sidewall x     {wall0:8.2f} -> {wall1:8.2f} nm   "
          f"({wall0 - wall1:+.2f} of film)")
    print(f"   mask top       {top0:8.2f} -> {top1:8.2f} nm   "
          f"({top1 - top0:+.2f})")
    print("\n   wrote polymer_initial/_final .vtp and _volume.vtu")


if __name__ == "__main__":
    main()
