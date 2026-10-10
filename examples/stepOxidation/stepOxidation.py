#!/usr/bin/env python3
"""
Step (Half-Fin) Oxidation Example (ViennaPS)
============================================
Simulates thermal oxidation of a silicon step geometry modelled as a half-fin.
The reflective boundary at x = 0 represents the centre of a symmetric fin;
the step wall is at x = finWidth / 2.  In the visible simulation domain
[0, xExtent] the raised platform occupies x in [0, finWidth/2] and the flat
substrate occupies x in [finWidth/2, xExtent].

Oxide grows on:
    · the top surface of the raised platform (y = finHeight)
    · the step wall (x = finWidth/2, y in [0, finHeight])
    · the flat substrate around the fin (y = 0)

Usage:
    python stepOxidation.py [config.txt]

All lengths are in micrometers, time in hours, pressure in atm.
"""

import sys
import time
import viennaps as ps

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
cfg = ps.readConfigFile(config_file)

ps.setDimension(int(cfg["dimensions"]))
ps.setNumThreads(int(cfg["numThreads"]))
ps.Logger.setLogLevel(ps.LogLevel.ERROR)

oxide_thickness = cfg["oxideThickness"]
oxidation_time = cfg["oxidationTime"]
temperature = cfg["temperature"]
pressure = cfg["pressure"]
oxidant = cfg["oxidant"]
orientation = cfg["orientation"]
output_prefix = cfg["outputPrefix"]

# ── Build Si half-fin (step) geometry ────────────────────────────────────────
# MakeFin with halfFin=True calls halveXAxis(), clipping the domain to
# [0, xExtent].  The fin occupies x in [0, finWidth/2]; the step wall is
# at x = finWidth/2.
domain = ps.Domain(
    gridDelta=cfg["gridDelta"],
    xExtent=2.0 * cfg["xExtent"],
    yExtent=2.0 * cfg.get("zExtent", cfg["xExtent"] / 2),
    boundary=ps.BoundaryType.REFLECTIVE_BOUNDARY,
)
ps.MakeFin(
    domain,
    finWidth=cfg["finWidth"],
    finHeight=cfg["finHeight"],
    halfFin=True,
).apply()

# ── Oxide seed ────────────────────────────────────────────────────────────────
seed_thickness = max(oxide_thickness, cfg["gridDelta"])
domain.duplicateTopLevelSet(ps.Material.SiO2)
ps.Process(domain, ps.SphereDistribution(radius=seed_thickness)).apply()

# ── Oxidation model ───────────────────────────────────────────────────────────
model = ps.Oxidation()
model.setTemperature(temperature)
model.setTime(oxidation_time)
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(str(int(orientation)))
model.setInitialOxideThickness(seed_thickness)

model.setGpuMode(cfg.get("useGpu", "cpu"))
model.setGpuPreconditioner(cfg.get("gpuPreconditioner", "jacobi"))

if cfg.get("maxGridPoints", 0) > 0:
    model.setMaxGridPoints(int(cfg["maxGridPoints"]))

model.saveSurfaceMesh(domain, output_prefix + "_initial.vtp")

t0 = time.perf_counter()
ps.Process(domain, model, 0.0).apply()
elapsed_sim = time.perf_counter() - t0

model.saveSurfaceMesh(domain, output_prefix + "_after.vtp")
model.saveVolumeMesh(domain, output_prefix + "_after")

print(f"Simulation time: {elapsed_sim:.2f} s")
print(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{model.estimatePlanarOxideThickness(seed_thickness):.4f} µm oxide."
)
print(
    f"Wrote {output_prefix}_initial.vtp, {output_prefix}_after.vtp, "
    f"and {output_prefix}_after_volume.vtu"
)
