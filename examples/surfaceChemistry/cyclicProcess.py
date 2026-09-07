#!/usr/bin/env python3
"""A cyclic process: two chemistries, coverages carried from phase to phase.

The Python counterpart of cyclicProcess.cpp, through the same reader and the
same model. A cycle is four phases, and two of them use a different chemistry
from the other two, so the process is two reaction files and a phase list:

    dose        the dose mechanism, its traced species flowing
    purge       the same chemistry, nothing flowing
    co-reactant the co-reactant mechanism, its traced species flowing
    purge       the same chemistry, nothing flowing

Each pulse flows exactly the species its own mechanism traces, so the driver
names no species and works for any pair of half-cycle reaction files. The
coverages carry from each phase into the next: what the dose leaves on the
surface is what the co-reactant acts on, which is the whole content of a
cycle and is why the coverages here are integrated in time.

    python3 cyclicProcess.py --cycles 25
    python3 cyclicProcess.py --cycles 25 --engine cpu
    python3 cyclicProcess.py --dose-file reactions/al2o3_tma.mechanism.json \\
        --coreactant-file reactions/al2o3_h2o.mechanism.json \\
        --film Al2O3 --initial O=1.0 --initial Al=1.0

Growing into a deep trench is the point of running this on a geometry: the
conformality of each pulse follows from the sticking its reaction file
declares.

Given a .yaml this compiles it with ViennaChem; given a .mechanism.json, or a
.yaml with ViennaChem missing, it reads the mechanism data with the same C++
reader the C++ example uses, so it runs standalone.

--lateral builds the same lateral high aspect ratio cavity the C++ driver
does, from the level-set primitives, so the two drivers cover the same
geometries.
"""

import argparse
import json
import os
import pathlib
import sys

import viennaps as ps
import viennals as ls

HERE = pathlib.Path(os.path.abspath(__file__)).parent
REACTIONS = HERE / "reactions"


def make_t(grid, opening_depth, opening_width, gap_length, gap_height,
           x_pad, material):
    """The lateral high aspect ratio cavity, in 2-D.

    A direct translation of makeT in examples/atomicLayerDeposition/
    geometry.hpp, which the C++ driver uses: a planar substrate with a
    vertical opening and a horizontal gap cut out of it. Only the left half is
    built, the reflective boundary at x = 0 supplying the other.

    The domain is constructed on the cavity's own bounds rather than
    default-constructed, since a domain carries its extent and a default one
    has none to insert a level set into.
    """
    bounds = [0.0, opening_width / 2.0 + x_pad + gap_length,
              -grid, opening_depth + gap_height + grid]
    boundary = [ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
                ls.BoundaryConditionEnum.INFINITE_BOUNDARY]
    domain = ps.Domain(bounds, boundary, grid)

    substrate = ls.Domain(bounds, boundary, grid)
    ls.MakeGeometry(
        substrate,
        ls.Plane([0.0, opening_depth + gap_height], [0.0, 1.0])).apply()
    domain.insertNextLevelSetAsMaterial(substrate, material)

    # the vertical opening the precursor enters through
    vert = ls.Domain(domain.getGrid())
    ls.MakeGeometry(vert, ls.Box(
        [-grid, 0.0],
        [opening_width / 2.0, gap_height + opening_depth + grid])).apply()
    domain.applyBooleanOperation(
        vert, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    # the horizontal cavity whose conformality is measured
    hori = ls.Domain(domain.getGrid())
    ls.MakeGeometry(hori, ls.Box(
        [opening_width / 2.0 - grid, 0.0],
        [opening_width / 2.0 + gap_length, gap_height])).apply()
    domain.applyBooleanOperation(
        hori, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)
    return domain


def traced_labels(path):
    """The species a pulse of this mechanism flows.

    Every traced gas species carrying a label, which is the same rule the C++
    driver applies. Ions live in their own block and are not phase-gated."""
    with open(path) as f:
        data = json.load(f)
    return [g["label"] for g in data.get("gas", [])
            if g.get("traced") and g.get("label")]


def load(spec):
    """A mechanism from a .yaml or .mechanism.json path, or a bare stem."""
    given = pathlib.Path(spec)
    stem = given.stem
    if stem.endswith(".mechanism"):
        stem = stem[: -len(".mechanism")]
    base = given.parent if given.parent != pathlib.Path(".") else REACTIONS
    yaml = base / f"{stem}.yaml"
    compiled = base / f"{stem}.mechanism.json"
    try:
        import viennachem as vc
    except ImportError:
        if not compiled.exists():
            sys.exit(f"neither ViennaChem nor {compiled.name} is available")
        print(f"  {compiled.name} (ViennaChem is not installed)")
        return ps.ChemicalMechanism.fromFile(str(compiled)), str(compiled)

    if compiled.exists() and compiled.stat().st_mtime < yaml.stat().st_mtime:
        print(f"  {yaml.name} is newer than its mechanism data; recompiling")
    if not yaml.exists():
        if not compiled.exists():
            sys.exit(f"neither {yaml.name} nor {compiled.name} exists")
        print(f"  {compiled.name}")
        return ps.ChemicalMechanism.fromFile(str(compiled)), str(compiled)
    data = vc.from_file(str(yaml))
    vc.write(str(compiled), data)
    print(f"  {yaml.name}")
    return ps.ChemicalMechanism.fromJSON(json.dumps(data)), str(compiled)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cycles", type=int, default=25)
    p.add_argument("--dose", type=float, default=0.2, help="s")
    p.add_argument("--purge", type=float, default=3.0, help="s")
    p.add_argument("--coreactant", type=float, default=15.0, help="s")
    # Coverage sub-steps per pulse. Every one re-traces the fluxes, because
    # the sticking depends on the coverage, so this is what a cycle costs.
    p.add_argument("--dose-steps", type=int, default=20)
    p.add_argument("--coreactant-steps", type=int, default=30)
    p.add_argument("--max-change", type=float, default=1e-3,
                   help="largest coverage change per integration sub-step")
    p.add_argument("--width", type=float, default=60.0, help="nm")
    p.add_argument("--depth", type=float, default=300.0, help="nm")
    p.add_argument("--grid", type=float, default=1.0, help="nm")
    p.add_argument("--rays", type=int, default=1000)
    p.add_argument("--engine", choices=("auto", "cpu", "gpu"), default="auto",
                   help="flux engine; auto uses the GPU when one is available")
    p.add_argument("--dose-file",
                   default="reactions/al2o3_tma.mechanism.json",
                   help="the mechanism the dose pulse uses")
    p.add_argument("--coreactant-file",
                   default="reactions/al2o3_h2o.mechanism.json",
                   help="the mechanism the co-reactant pulse uses")
    p.add_argument("--film", default="Al2O3",
                   help="material of the film that is deposited")
    p.add_argument("--name", default="cyclicProcess",
                   help="process name recorded in the output")
    p.add_argument("--initial", action="append", default=[],
                   metavar="SPECIES=FRACTION",
                   help="coverage a species holds before the first cycle; "
                        "repeatable")
    p.add_argument("--intermediate", action="store_true",
                   help="log the intermediate surfaces of each cycle")
    p.add_argument("--lateral", action="store_true",
                   help="a lateral high aspect ratio cavity instead of a "
                        "trench; switches the length unit to micrometres")
    p.add_argument("--gap-length", type=float, default=100.0, help="um")
    p.add_argument("--gap-height", type=float, default=0.5, help="um")
    p.add_argument("--opening-width", type=float, default=10.0, help="um")
    p.add_argument("--opening-depth", type=float, default=0.5, help="um")
    p.add_argument("--x-pad", type=float, default=0.5, help="um")
    p.add_argument("--out", default="cyclicProcess")
    return p.parse_args()


def main():
    o = parse()
    ps.setDimension(2)
    if o.intermediate:
        ps.Logger.setLogLevel(ls.LogLevel.INTERMEDIATE)
    # the trench is quoted in nanometres and the lateral cavity in
    # micrometres, and every rate constant is converted to the active unit
    ps.Length.setUnit("um" if o.lateral else "nm")
    ps.Time.setUnit("s")

    print("reaction files:")
    dose, dose_json = load(o.dose_file)
    coreactant, coreactant_json = load(o.coreactant_file)

    if o.lateral:
        domain = make_t(o.grid, o.opening_depth, o.opening_width, o.gap_length,
                        o.gap_height, o.x_pad, ps.Material.Si)
    else:
        domain = ps.Domain(gridDelta=o.grid, xExtent=200.0, yExtent=400.0)
        ps.MakeTrench(domain=domain, trenchWidth=o.width, trenchDepth=o.depth,
                      trenchTaperAngle=0.0, maskHeight=0.0, maskTaperAngle=0.0,
                      halfTrench=False, material=ps.Material.Si,
                      maskMaterial=ps.Material.Mask).apply()
    domain.duplicateTopLevelSet(getattr(ps.Material, o.film))
    domain.saveSurfaceMesh(f"{o.out}_initial.vtp")

    model = ps.SurfaceChemistry()
    model.addMechanism("dose", dose)
    model.addMechanism("coreactant", coreactant)
    model.setAtomicLayerProcess()
    model.setMaxCoverageChange(o.max_change)
    for entry in o.initial:
        if "=" not in entry:
            sys.exit(f"--initial expects SPECIES=FRACTION, got {entry!r}")
        name, value = entry.split("=", 1)
        model.setInitialCoverage(name, float(value))
    model.setProcessName(o.name)

    # The species each pulse flows. A purge names none, so only the thermal
    # steps of that half-cycle's chemistry run through it.
    alp = ps.AtomicLayerProcessParameters()
    alp.numCycles = o.cycles
    alp.addPhase("dose", o.dose, o.dose / o.dose_steps,
                 traced_labels(dose_json), "dose")
    alp.addPhase("purge_dose", o.purge, o.purge / 4.0, [], "dose")
    alp.addPhase("coreactant", o.coreactant, o.coreactant / o.coreactant_steps,
                 traced_labels(coreactant_json), "coreactant")
    alp.addPhase("purge_coreactant", o.purge, o.purge / 4.0, [], "coreactant")

    print(f"\n{o.name}: {o.cycles} cycles of {o.dose} s dose / "
          f"{o.purge} s purge / {o.coreactant} s co-reactant / "
          f"{o.purge} s purge")
    if o.lateral:
        print(f"lateral cavity {o.gap_length} um long, {o.gap_height} um high "
              f"(aspect ratio {o.gap_length / o.gap_height:.0f})\n")
    else:
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

    per_angstrom = 1.0e4 if o.lateral else 10.0
    print(f"\ngrowth per cycle on the open field = "
          f"{model.growthPerCycle() * per_angstrom:.4f} A/cycle")

    domain.saveSurfaceMesh(f"{o.out}_final.vtp", True)
    domain.saveVolumeMesh(f"{o.out}_final")
    print(f"\nwrote {o.out}_initial.vtp, {o.out}_final.vtp and "
          f"{o.out}_final_volume.vtu\n"
          "  open the .vtu in ParaView and colour by 'Material' to see the "
          "film against the substrate")


if __name__ == "__main__":
    main()
