#!/usr/bin/env python3
"""The two arms in three dimensions, from Python: the same trench, the same
mechanism file, a level set advected by ViennaPS against a voxel grid moved
by VoxelProcess. Both arms run on the CPU or the GPU, so the whole
2x2 matrix of the benchmark is four invocations of this script:

    python voxelComparison3D.py
    python voxelComparison3D.py --voxel-gpu
    python voxelComparison3D.py --ls-gpu --voxel-gpu
    python voxelComparison3D.py --ls-gpu
"""

import argparse
import os
import time

import viennaps as ps
import viennals
import viennals.d3 as ls3

ps.setDimension(3)

HERE = os.path.dirname(os.path.abspath(__file__))
MECHANISMS = os.path.join(HERE, "..", "surfaceChemistry", "reactions")

W, DEPTH, GD = 20.0, 30.0, 1.0
MASK_H = 15.0


def make_domain():
    domain = ps.Domain(gridDelta=GD, xExtent=4 * W, yExtent=4 * W)
    # a FLAT substrate: the slot in the mask is the only opening, and the
    # etch digs the trench itself, as a real masked etch does
    ps.MakeTrench(domain=domain, trenchWidth=W, trenchDepth=0.0,
                  maskHeight=MASK_H, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    return domain


def band_top(nodes, lo, hi, axis=0):
    """Highest point with material inside a lateral band: the shared probe."""
    best = None
    for n in nodes:
        if lo <= n[axis] <= hi:
            best = n[2] if best is None else max(best, n[2])
    return best


def surface_nodes(domain):
    mesh = viennals.Mesh()  # top-level Mesh, dimension-independent
    ls3.ToSurfaceMesh(domain.getLevelSets()[-1], mesh).apply()
    return mesh.getNodes()


def level_set_arm(mech, duration, gpu, rays=200):
    domain = make_domain()
    before_field = band_top(surface_nodes(domain), -0.45 * 4 * W, -0.30 * 4 * W)
    before_floor = band_top(surface_nodes(domain), -0.35 * W / 2, 0.35 * W / 2)
    domain.saveSurfaceMesh(filename=mech.name + "_3d_ls_initial.vtp")

    process = ps.Process(domain, ps.SurfaceChemistry(mech), duration)
    process.setFluxEngineType(ps.FluxEngineType.GPU_TRIANGLE if gpu
                              else ps.FluxEngineType.CPU_TRIANGLE)
    rt = ps.RayTracingParameters()
    rt.raysPerPoint = rays
    process.setParameters(rt)
    cov = ps.CoverageParameters()
    cov.tolerance = 1e-6
    cov.maxIterations = 40
    process.setParameters(cov)

    process.apply()
    pt = process.getProcessingTimes()

    domain.saveSurfaceMesh(filename=mech.name + "_3d_ls_final.vtp")
    nodes = surface_nodes(domain)
    mask = band_top(nodes, -0.45 * 4 * W, -0.30 * 4 * W) - before_field
    floor = band_top(nodes, -0.35 * W / 2, 0.35 * W / 2) - before_floor
    return mask, floor, pt


def voxel_probe(vox, lo, hi):
    """The voxel version of the same probe: per column, the top of the
    topmost cell holding material, fill-weighted; averaged over the band."""
    dims, corner, delta = vox.dims(), vox.minCorner(), vox.gridDelta()
    fills = vox.fills()
    total, count = 0.0, 0
    for i in range(dims[0]):
        x = corner[0] + delta * (i + 0.5)
        if x < lo or x > hi:
            continue
        for j in range(dims[1]):
            for k in reversed(range(dims[2])):
                cid = vox.cellId([i, j, k])
                if cid < 0 or fills[cid] <= 1e-6:
                    continue
                total += corner[2] + delta * (k + 1) - (1 - fills[cid]) * delta
                count += 1
                break
    return total / count if count else 0.0


def voxel_arm(mech, duration, steps, gpu, rays=500000):
    domain = make_domain()
    vox = ps.VoxelProcess(domain, mech, depthBelow=DEPTH + 6.0,
                          coverAbove=MASK_H + 4.0)
    vox.setRaysPerStep(rays)
    vox.setNormalEstimator(ps.NormalEstimator.FillGradientYoungs)
    vox.setTraversalEngine(ps.TraversalEngine.EmbreeBVH)
    vox.setUseGPU(gpu)

    m0 = voxel_probe(vox, -0.45 * 4 * W, -0.30 * 4 * W)
    f0 = voxel_probe(vox, -0.35 * W / 2, 0.35 * W / 2)
    vox.writeCells(mech.name + "_3d_voxel_initial.vtu")

    report = vox.apply(duration, steps)

    vox.writeCells(mech.name + "_3d_voxel_final.vtu")
    mask = voxel_probe(vox, -0.45 * 4 * W, -0.30 * 4 * W) - m0
    floor = voxel_probe(vox, -0.35 * W / 2, 0.35 * W / 2) - f0
    return mask, floor, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ls-gpu", action="store_true")
    p.add_argument("--voxel-gpu", action="store_true")
    p.add_argument("--time-factor", type=float, default=15.0,
                   help="multiples of the ~2 nm reference dose")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--mechanism", default=os.path.join(
        MECHANISMS, "sf6o2.mechanism.json"))
    args = p.parse_args()

    ps.setNumThreads(12)
    ps.Length.setUnit("nm")
    ps.Time.setUnit("s")
    ps.Logger.setLogLevel(ps.LogLevel.WARNING)

    mech = ps.ChemicalMechanism.fromFile(args.mechanism)
    gamma = mech.sourceFluxes("")
    theta = mech.solveCoverages(gamma, [0.0] * len(mech.coverageNames))
    analytic = mech.growthRate(gamma, theta)
    duration = args.time_factor * abs(2.0 / analytic)

    print(f"3D trench: {mech.name}  (analytic {analytic:.4g} nm/s, "
          f"{duration:.3g} s)")

    mask, floor, pt = level_set_arm(mech, duration, args.ls_gpu)
    print(f"  level set ({'GPU' if args.ls_gpu else 'CPU'}):  "
          f"mask {mask:+.3f}  floor {floor:+.3f}   [{pt.total:.1f} s: "
          f"flux {pt.flux:.1f}, advection {pt.advection:.1f}, other "
          f"{pt.total - pt.flux - pt.advection:.1f}]")

    vmask, vfloor, r = voxel_arm(mech, duration, args.steps, args.voxel_gpu)
    total = (r.secondsTransport + r.secondsChemistry + r.secondsAdvance +
             r.secondsRelabel)
    print(f"  voxel     ({'GPU' if args.voxel_gpu else 'CPU'}):  "
          f"mask {vmask:+.3f}  floor {vfloor:+.3f}   [{total:.1f} s: "
          f"transport {r.secondsTransport:.1f}, chemistry "
          f"{r.secondsChemistry:.1f}, advance {r.secondsAdvance:.1f}, "
          f"relabel {r.secondsRelabel:.1f}]")
    print(f"  floor, voxel against level set: "
          f"{100 * (vfloor / floor - 1):.1f}%")


if __name__ == "__main__":
    main()
