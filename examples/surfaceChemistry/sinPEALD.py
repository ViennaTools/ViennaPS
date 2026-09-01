#!/usr/bin/env python3
"""PE-ALD of SiNx from diiodosilane and an N2/H2 plasma, in a trench.

The same process as sinPEALD.cpp, through the same reader and the same model.
A cycle is four steps, and two of them use a different chemistry from the
other two, so the process is two reaction files and a phase list:

    dose    reactions/sin_peald_dis_dose.yaml      SiH2I2 flowing
    purge   the same chemistry, nothing flowing
    plasma  reactions/sin_peald_n2h2_plasma.yaml   N, H, NH, N2+ flowing
    purge   the same chemistry, nothing flowing

The coverages carry from each step into the next: what the dose leaves on the
surface is what the plasma acts on, which is the whole content of an ALD
cycle and is why the coverages here are integrated in time.

    python sinPEALD.py --cycles 50                   # GPU if one is present
    python sinPEALD.py --cycles 50 --engine cpu      # force the CPU engine
    python sinPEALD.py --width 60 --depth 400        # aspect ratio 6.7

Growing into a deep trench is the point of running this on a geometry: the
conformality of the plasma step follows from the recombination probabilities
of the N and H radicals, which the reaction files carry as the sticking of
their last adsorption step.

    M. Zeghouane et al., Mater. Sci. Semicond. Process. 184 (2024) 108851.

Given a .yaml this compiles it with ViennaChem; given a .mechanism.json, or a
.yaml with ViennaChem missing, it reads the mechanism data with the same C++
reader the C++ example uses, so it runs standalone.
"""

import argparse
import json
import os
import pathlib
import sys

import viennaps as ps

HERE = pathlib.Path(os.path.abspath(__file__)).parent
REACTIONS = HERE / "reactions"


def load(stem):
    """A mechanism from `stem`.yaml, compiling it if ViennaChem is present."""
    yaml = REACTIONS / f"{stem}.yaml"
    compiled = REACTIONS / f"{stem}.mechanism.json"
    try:
        import viennachem as vc
    except ImportError:
        if not compiled.exists():
            sys.exit(f"neither ViennaChem nor {compiled.name} is available")
        print(f"  {compiled.name} (ViennaChem is not installed)")
        return ps.ChemicalMechanism.fromFile(str(compiled))

    if compiled.exists() and compiled.stat().st_mtime < yaml.stat().st_mtime:
        print(f"  {yaml.name} is newer than its mechanism data; recompiling")
    data = vc.from_file(str(yaml))
    vc.write(str(compiled), data)
    print(f"  {yaml.name}")
    return ps.ChemicalMechanism.fromJSON(json.dumps(data))


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cycles", type=int, default=25)
    p.add_argument("--dose", type=float, default=0.2, help="s")
    p.add_argument("--purge", type=float, default=3.0, help="s")
    p.add_argument("--plasma", type=float, default=15.0, help="s")
    # Coverage sub-steps per pulse. Every one re-traces the fluxes, because
    # the sticking depends on the coverage, so this is what a cycle costs.
    p.add_argument("--dose-steps", type=int, default=20)
    p.add_argument("--plasma-steps", type=int, default=30)
    p.add_argument("--max-change", type=float, default=1e-3,
                   help="largest coverage change per integration sub-step")
    p.add_argument("--width", type=float, default=60.0, help="nm")
    p.add_argument("--depth", type=float, default=300.0, help="nm")
    p.add_argument("--grid", type=float, default=1.0, help="nm")
    p.add_argument("--rays", type=int, default=1000)
    p.add_argument("--engine", choices=("auto", "cpu", "gpu"), default="auto",
                   help="flux engine; auto uses the GPU when one is available")
    p.add_argument("--out", default="sinPEALD")
    return p.parse_args()


def main():
    o = parse()
    ps.setDimension(2)
    ps.Length.setUnit("nm")
    ps.Time.setUnit("s")

    print("reaction files:")
    dose = load("sin_peald_dis_dose")
    plasma = load("sin_peald_n2h2_plasma")

    domain = ps.Domain(gridDelta=o.grid, xExtent=200.0, yExtent=400.0)
    ps.MakeTrench(domain=domain, trenchWidth=o.width, trenchDepth=o.depth,
                  trenchTaperAngle=0.0, maskHeight=0.0, maskTaperAngle=0.0,
                  halfTrench=False, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    domain.duplicateTopLevelSet(ps.Material.SiN)
    domain.saveSurfaceMesh(f"{o.out}_initial.vtp")

    model = ps.SurfaceChemistry()
    model.addMechanism("dose", dose)
    model.addMechanism("plasma", plasma)
    model.setAtomicLayerProcess()
    model.setMaxCoverageChange(o.max_change)
    model.setProcessName("sinPEALD")

    # The species each pulse flows. A purge names none, so only the thermal
    # steps of that half-cycle's chemistry run through it.
    alp = ps.AtomicLayerProcessParameters()
    alp.numCycles = o.cycles
    alp.addPhase("dose", o.dose, o.dose / o.dose_steps,
                 ["SiH2I2_flux"], "dose")
    alp.addPhase("purge_dose", o.purge, o.purge / 4.0, [], "dose")
    alp.addPhase("plasma", o.plasma, o.plasma / o.plasma_steps,
                 ["N_flux", "H_flux", "NH_flux"], "plasma")
    alp.addPhase("purge_plasma", o.purge, o.purge / 4.0, [], "plasma")

    print(f"\nSiNx PE-ALD: {o.cycles} cycles of {o.dose} s dose / "
          f"{o.purge} s purge / {o.plasma} s plasma / {o.purge} s purge")
    print(f"trench {o.width} nm wide, {o.depth} nm deep "
          f"(aspect ratio {o.depth / o.width:.1f})\n")

    # Only the transport moves to the device; the coverage integration runs
    # on the host either way, so the two engines agree to within ray noise.
    have_gpu = ps.gpuAvailable()
    if o.engine == "gpu" and not have_gpu:
        sys.exit("no GPU available: build with VIENNAPS_USE_GPU=ON, or use "
                 "--engine cpu")
    use_gpu = o.engine == "gpu" or (o.engine == "auto" and have_gpu)
    print(f"flux engine: {'GPU' if use_gpu else 'CPU'}"
          f"{' (auto)' if o.engine == 'auto' else ''}")

    process = ps.Process(domain, model)
    process.setFluxEngineType(ps.FluxEngineType.GPU_LINE if use_gpu
                              else ps.FluxEngineType.CPU_DISK)
    process.setParameters(alp)
    tracing = ps.RayTracingParameters()
    tracing.raysPerPoint = o.rays
    process.setParameters(tracing)
    process.apply()

    print(f"\ngrowth per cycle on the open field = "
          f"{model.growthPerCycle() * 10:.4f} A/cycle"
          f"   (measured 0.36-0.40 A/cycle at 300 C)")

    domain.saveSurfaceMesh(f"{o.out}_final.vtp", True)
    domain.saveVolumeMesh(f"{o.out}_final")
    print(f"\nwrote {o.out}_initial.vtp, {o.out}_final.vtp and "
          f"{o.out}_final_volume.vtu\n"
          "  open the .vtu in ParaView and colour by 'Material' to see the "
          "film against the substrate")


if __name__ == "__main__":
    main()
