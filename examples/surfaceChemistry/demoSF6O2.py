#!/usr/bin/env python3
"""SF6/O2 from a reaction file, beside the model it was written from.

`reactions/sf6o2.yaml` is ViennaPS's own SF6/O2 silicon etch -- `psSF6O2Etching`
plus `psPlasmaEtching` -- written as reactions and rate constants, with every
number taken from `SF6O2Etching::defaultParameters()`. This etches the same
masked trench with both and writes both meshes, so the two profiles can be laid
over each other.

The two apply the sticking differently: the reference multiplies it into the
particle's re-emission alone, while a reaction file's sticking enters the rate
law as well. The two coincide at beta = 1, so that is where the comparison is
run; `ViennaChem/tests/test_sf6o2.py` reports what the model's own beta = 0.7
costs.

    python demoSF6O2.py [--gpu]
"""

import argparse
import json

import numpy as np
import viennals
import viennals.d2 as ls2
import viennaps as ps

from surfaceChemistry import REACTIONS, load

TIME = 2.0
RAYS = 3000
ENGINE = ps.FluxEngineType.CPU_DISK   # replaced by --gpu

ps.setDimension(2)
ps.setNumThreads(16)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)


def reference(beta=1.0):
    """ViennaPS's hand-written model, at its defaults but for the sticking."""
    model = ps.SF6O2Etching()
    p = model.getParameters()
    for material in (ps.Material.Si, ps.Material.Mask):
        p.beta_E.set(material, beta)
        p.beta_P.set(material, beta)
    model.setParameters(p)
    return model


def generated():
    """The same chemistry, built from the reaction file."""
    data = load(REACTIONS / "sf6o2.yaml", None)[0]
    return ps.SurfaceChemistry(data)


def matched_generated():
    """The reaction file with both stickings at 1, where the conventions meet."""
    import viennachem as vc
    data = vc.from_file(str(REACTIONS / "sf6o2.yaml"))
    for g in data["gas"]:
        if g["sticking"] is not None:
            g["sticking"]["s0"] = 1.0
    for rx in data["reactions"]:
        if rx["constant"]["kind"] == "sticking":
            rx["constant"]["s0"] = 1.0
    return ps.SurfaceChemistry(ps.ChemicalMechanism.fromJSON(json.dumps(data)))


def nodes(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def etch(model, stem):
    dom = ps.Domain(gridDelta=1.0, xExtent=200.0, yExtent=200.0)
    ps.MakeTrench(domain=dom, trenchWidth=60.0, trenchDepth=40.0,
                  maskHeight=30.0, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    before = nodes(dom)
    dom.saveSurfaceMesh(filename=f"{stem}_initial.vtp", addInterfaces=True)
    dom.saveVolumeMesh(f"{stem}_initial")

    p = ps.Process(dom, model, TIME)
    p.setFluxEngineType(ENGINE)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 30
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = RAYS
    p.setParameters(r)
    p.apply()

    after = nodes(dom)
    dom.saveSurfaceMesh(filename=f"{stem}_final.vtp", addInterfaces=True)
    dom.saveVolumeMesh(f"{stem}_final")

    depth = before[:, 1].min() - after[:, 1].min()
    mask = after[:, 1].max() - before[:, 1].max()
    mid = 0.5 * (after[:, 1].min() + before[:, 1].min())
    band = after[np.abs(after[:, 1] - mid) < 4.0]
    wall = band[band[:, 0] > 0][:, 0].min() if len(band[band[:, 0] > 0]) else np.nan
    return depth, mask, wall


def main():
    global ENGINE
    ap = argparse.ArgumentParser(description="SF6/O2, generated against reference.")
    ap.add_argument("--gpu", action="store_true", help="trace on the device")
    if ap.parse_args().gpu:
        ENGINE = ps.FluxEngineType.GPU_TRIANGLE

    print("SF6/O2: a reaction file against the model it was written from\n")
    engine = "GPU" if ENGINE != ps.FluxEngineType.CPU_DISK else "CPU"
    print(f"  masked trench, {TIME:g} s, {RAYS} rays per point, {engine}\n")
    print(f"  {'':22s} {'depth [nm]':>11s} {'mask [nm]':>10s} {'wall x [nm]':>12s}")

    results = {}
    for label, model in (("hand-written model", reference(1.0)),
                         ("from sf6o2.yaml", matched_generated())):
        stem = "sf6o2_" + ("reference" if "hand" in label else "generated")
        d, m, w = etch(model, stem)
        results[label] = (d, m, w)
        print(f"  {label:22s} {d:11.3f} {m:10.3f} {w:12.3f}")

    (d0, m0, w0), (d1, m1, w1) = results.values()
    print(f"\n  difference             {abs(d1 - d0) / d0 * 100:10.2f}% "
          f"{abs(m1 - m0):9.3f}  {abs(w1 - w0):11.3f}")
    print("\n  wrote, for each of the two:  _initial.vtp / _initial_volume.vtu")
    print("                               _final.vtp   / _final_volume.vtu")
    print("  The initial surface is the trench both started from, so the two")
    print("  final profiles can be laid over it in ParaView, coloured by")
    print("  Material.")


if __name__ == "__main__":
    main()
