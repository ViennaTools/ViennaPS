#!/usr/bin/env python3
"""One model, many chemistries: reactions in, simulation out.

Every mechanism in `reactions/` runs through THIS script, and through the C++
program beside it. Nothing here is specific to a chemistry: the reaction file
decides whether the surface grows or is etched, which particles are traced, how
many coverages there are, what the rate laws look like, and how the chemistry
differs from one material to the next.

    python surfaceChemistry.py reactions/silane.yaml
    python surfaceChemistry.py reactions/sf6o2.yaml --thickness 30
    python surfaceChemistry.py reactions/diamond.yaml -D 3 --gpu

What happens:

    reaction file  ->  front end (parse, balance, infer free sites, nu, SymPy)
                   ->  mechanism data, the `.ir.json`
                   ->  ps.ChemicalMechanism
                   ->  ps.SurfaceChemistry, the generic ViennaPS model
                   ->  a feature-scale simulation

No chemistry-specific C++ is written or compiled at any point.

Given a `.yaml` this compiles it with ViennaChem. Given a `.mechanism.json` --
or a `.yaml` with ViennaChem not installed -- it reads the mechanism data with
the same C++ reader the C++ example uses, so it runs standalone.
"""

import argparse
import json
import os
import pathlib
import sys

import viennaps as ps

HERE = os.path.dirname(os.path.abspath(__file__))
REACTIONS = pathlib.Path(HERE) / "reactions"   # the reaction files ship here

# ViennaChem compiles a reaction file into mechanism data. It is a separate
# library (`pip install viennachem`) and is needed only to read a `.yaml`:
# mechanism data is readable without it, by the C++ reader below.
try:
    import viennachem as vc
    HAVE_FRONTEND = True
except ImportError:
    HAVE_FRONTEND = False


def parse_args():
    p = argparse.ArgumentParser(
        prog="surfaceChemistry.py",
        description="Run a reaction file in ViennaPS: one model, any chemistry.",
    )
    p.add_argument("reactions", nargs="?",
                   default=os.path.join(HERE, "reactions", "silane.yaml"),
                   help="the reaction file (.yaml, or .ir.json)")
    p.add_argument("-D", "--dim", type=int, default=2, choices=(2, 3),
                   help="2 = trench, 3 = cylindrical hole")
    p.add_argument("-T", "--temperature", type=float, default=None,
                   help="override the temperature in the reaction file [K]")
    p.add_argument("--thickness", type=float, default=20.0,
                   help="target thickness grown or removed on the field [nm]")
    p.add_argument("--time", type=float, default=None,
                   help="process time [s]; overrides --thickness")
    p.add_argument("--width", type=float, default=80.0, help="feature width [nm]")
    p.add_argument("--depth", type=float, default=120.0, help="feature depth [nm]")
    p.add_argument("--mask", type=float, default=0.0,
                   help="mask height [nm]; 0 for no mask. An etch needs one to "
                        "show selectivity")
    p.add_argument("--grid", type=float, default=2.0, help="grid spacing [nm]")
    p.add_argument("--rays", type=int, default=1000, help="rays per point")
    p.add_argument("--gpu", action="store_true", help="use the GPU flux engine")
    p.add_argument("--no-volume", dest="volume", action="store_false",
                   help="write only the surface mesh, no meshed volume")
    p.add_argument("--material", default="Si", help="substrate material name")
    p.add_argument("--film", default=None,
                   help="film material name (default: the solid species)")
    p.add_argument("-o", "--output", default=None, help="output file prefix")
    return p.parse_args()


def load(path, temperature):
    """The reaction file as a ps.ChemicalMechanism, by whichever route is open.

    A `.yaml` needs the front end. Mechanism data (`.ir.json`) does not: the C++
    loader reads it, which is how a ViennaPS checkout runs this with no Python
    toolchain installed.
    """
    path = str(path)
    data = (os.path.splitext(path)[0] + ".mechanism.json"
            if path.endswith(".yaml") else path)

    if path.endswith(".yaml") and HAVE_FRONTEND:
        compiled = vc.from_file(path)
        if temperature is not None:
            compiled["constants"]["temperature"] = temperature
        # one reader, handed the data in memory rather than a path
        return ps.ChemicalMechanism.fromJSON(json.dumps(compiled)), \
            "the reaction file"

    if not os.path.exists(data):
        sys.exit(f"no mechanism data at '{data}'. Either `pip install "
                 f"viennachem`, or compile it with "
                 f"`python -m viennachem {path} {data}`.")

    # Falling back for a .yaml means simulating DERIVED data, not the file that
    # was asked for. Silence here would run an edited reaction file's old
    # chemistry without a word, so say so, and refuse outright when the data is
    # demonstrably out of date.
    if path.endswith(".yaml"):
        print(f"note: ViennaChem is not installed, so '{os.path.basename(data)}'"
              f" is used instead of the reaction file itself.")
        if os.path.getmtime(path) > os.path.getmtime(data):
            sys.exit(f"'{path}' is NEWER than '{data}': the mechanism data is "
                     f"stale. Recompile it with "
                     f"`python -m viennachem {path} {data}`.")
    mech = ps.ChemicalMechanism.fromFile(data)
    if temperature is not None:
        mech.temperature = temperature
    return mech, os.path.basename(data)


def predict(mech, material=""):
    """The steady state and the surface velocity on one material.

    Everything comes from the mechanism, so this needs no front end: the ion
    yield channels are evaluated at normal incidence, which is what a blanket
    surface sees.
    """
    gamma = mech.sourceFluxes(material)
    theta = (mech.solveCoveragesOn(gamma, [0.0] * len(mech.coverageNames), material)
             if material else
             mech.solveCoverages(gamma, [0.0] * len(mech.coverageNames)))
    rate = (mech.growthRateOn(gamma, theta, material) if material
            else mech.growthRate(gamma, theta))
    return theta, rate


def report(mech, theta, rate, source):
    """What the model derived, before anything is simulated."""
    print(f"mechanism   : {mech.name}   (from {source})")
    print(f"temperature : {mech.temperature:.1f} K")
    print(f"solids      : {mech.solidNames}")
    print(f"coverages   : {mech.coverageNames}")
    particles = list(mech.particleLabels)
    if mech.hasIonSource:
        particles.append(f"[ion, {mech.ionMeanEnergy:g} eV]")
    print(f"particles   : {particles}")
    print("reactions   :")
    for eq in mech.reactionSummary():
        print(f"   {eq}")
    print("steady state:")
    for name, value in zip(mech.coverageNames, theta):
        print(f"   theta_{name:<8} = {value:.6e}")
    # a negative rate is an etch: the solid sits on the left of a reaction
    word = "etch rate  " if rate < 0 else "growth rate"
    print(f"   {word}  = {rate:.6e} nm/s  ({rate * 3600 / 1000:.4f} um/h)")


def save(domain, stem, volume=True):
    """The surface, and the meshed volume that renders as a solid body."""
    domain.saveSurfaceMesh(filename=f"{stem}.vtp", addInterfaces=True)
    if volume:
        domain.saveVolumeMesh(f"{stem}")  # writes <stem>_volume.vtu


def material(name, fallback):
    m = getattr(ps.Material, name, None)
    if m is None:
        print(f"unknown material '{name}', using {fallback}")
        m = getattr(ps.Material, fallback)
    return m


def main():
    args = parse_args()

    ps.setDimension(args.dim)
    ps.setNumThreads(16)
    ps.Length.setUnit("nm")
    ps.Time.setUnit("s")
    ps.Logger.setLogLevel(ps.LogLevel.INFO)

    # 1. the reaction file -> the mechanism -> the generic model
    mech, source = load(args.reactions, args.temperature)

    # the analytic estimate on a flat surface, which sets the process time
    theta, rate = predict(mech)
    report(mech, theta, rate, source)

    if rate == 0.0:
        sys.exit("the mechanism moves the surface nowhere; nothing to simulate")

    etching = rate < 0.0
    process_time = (args.time if args.time is not None
                    else args.thickness / abs(rate))
    prefix = args.output or mech.name

    # 2. geometry
    domain = ps.Domain(gridDelta=args.grid, xExtent=200.0, yExtent=200.0)
    if args.dim == 2:
        ps.MakeTrench(domain=domain, trenchWidth=args.width,
                      trenchDepth=args.depth, maskHeight=args.mask,
                      material=material(args.material, "Si"),
                      maskMaterial=ps.Material.Mask).apply()
    else:
        ps.MakeHole(domain=domain, holeRadius=args.width / 2.0,
                    holeDepth=args.depth, maskHeight=args.mask,
                    holeShape=ps.HoleShape.QUARTER,
                    material=material(args.material, "Si"),
                    maskMaterial=ps.Material.Mask).apply()
    # A deposition grows a new level set, named after the solid species. An etch
    # removes the substrate itself, so there is no film to add.
    if not etching:
        film = args.film or (mech.solidNames[0] if mech.solidNames else "PolySi")
        domain.duplicateTopLevelSet(material(film, "PolySi"))
    save(domain, f"{prefix}_{args.dim}D_initial", args.volume)

    # 3. run
    print(f"\nprocess time = {process_time:.3g} s for ~{args.thickness:g} nm "
          f"{'removed' if etching else 'of film'}")
    process = ps.Process(domain, ps.SurfaceChemistry(mech), process_time)
    process.setFluxEngineType(
        ps.FluxEngineType.GPU_LINE if args.gpu else ps.FluxEngineType.CPU_DISK
    )
    cov = ps.CoverageParameters()
    cov.tolerance = 1e-4
    cov.maxIterations = 20  # the delta metric floors on Monte-Carlo noise
    process.setParameters(cov)
    ray = ps.RayTracingParameters()
    ray.raysPerPoint = args.rays
    process.setParameters(ray)
    process.apply()

    save(domain, f"{prefix}_{args.dim}D_final", args.volume)
    print(f"wrote {prefix}_{args.dim}D_initial and _final "
          f"(.vtp surface, _volume.vtu meshed volume)")


if __name__ == "__main__":
    main()
