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
import viennaps as vps

vps.setDimension(2)

# ── Default parameters (match stepOxidation/config.txt) ─────────────────────
cfg = {
    "numThreads":      16,
    "gridDelta":       0.05,
    "xExtent":         1.0,
    "finWidth":        0.5,   # fin wall at x = finWidth/2 = 0.25 µm
    "finHeight":       1.0,   # step height above the substrate
    "oxideThickness":  0.0,
    "oxidationTime":   0.05,
    "timeStep":        0.01,
    "temperature":  1000.0,
    "pressure":        1.0,
    "oxidant":       "wet",
    "orientation":   "100",
    "maxGridPoints":     5000000,
    "outputPrefix":      "ps_step_oxidation",
    "useGpu":            "auto",   # auto | gpu | cpu
    "gpuPreconditioner": "jacobi", # jacobi | ilu0
}


def _parse_config(path: str) -> None:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                eq = line.find("=")
                if eq < 0:
                    continue
                key = line[:eq].strip()
                val = line[eq + 1:].split("#")[0].strip()
                if key in cfg:
                    t = type(cfg[key])
                    cfg[key] = t(val)
    except FileNotFoundError:
        pass


def _parse_oxidant(s: str):
    s = s.lower()
    if s in ("wet", "h2o"):
        return vps.OxidantType.Wet
    if s in ("dry", "o2"):
        return vps.OxidantType.Dry
    raise ValueError(f"Unknown oxidant '{s}'. Use wet/H2O or dry/O2.")


def _parse_orientation(s: str):
    s = s.lower()
    if s in ("100", "<100>", "si100"):
        return vps.SiliconOrientation.Si100
    if s in ("111", "<111>", "si111"):
        return vps.SiliconOrientation.Si111
    if s in ("poly", "polysi"):
        return vps.SiliconOrientation.PolySi
    raise ValueError(f"Unknown orientation '{s}'. Use 100, 111, or poly.")


# ── Config file ──────────────────────────────────────────────────────────────
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
_parse_config(config_file)

vps.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

grid_delta      = cfg["gridDelta"]
x_extent        = cfg["xExtent"]
fin_width       = cfg["finWidth"]
fin_height      = cfg["finHeight"]
oxide_thickness = cfg["oxideThickness"]
oxidation_time  = cfg["oxidationTime"]
time_step       = cfg["timeStep"]
temperature     = cfg["temperature"]
pressure        = cfg["pressure"]
oxidant         = _parse_oxidant(cfg["oxidant"])
orientation     = _parse_orientation(cfg["orientation"])
max_grid_points = cfg["maxGridPoints"]
output_prefix   = cfg["outputPrefix"]

# ── Build Si half-fin (step) geometry ────────────────────────────────────────
# Set up domain with bounds [-xExtent, xExtent], REFLECTIVE X, INFINITE Y.
# MakeFin with halfFin=True calls halveXAxis(), clipping the domain to
# [0, xExtent].  The fin occupies x in [0, finWidth/2]; the step wall is
# at x = finWidth/2.
domain = vps.Domain(vps.DomainSetup(grid_delta, 2.0 * x_extent, 0.0,
                                     vps.BoundaryType.REFLECTIVE_BOUNDARY))
vps.MakeFin(domain, fin_width, fin_height, 0.0, 0, 0, True).apply()

# ── Oxide seed ────────────────────────────────────────────────────────────────
import viennals as vls

seed_thickness = max(oxide_thickness, grid_delta)
ambient_ls = vls.Domain(domain.getLevelSets()[-1])
vls.GeometricAdvect(ambient_ls, vls.SphereDistribution(seed_thickness)).apply()
domain.insertNextLevelSetAsMaterial(ambient_ls, vps.Material.SiO2, False)

# ── Oxidation model ───────────────────────────────────────────────────────────
model = vps.Oxidation()
model.setTemperature(temperature)
model.setTime(oxidation_time)
model.setTimeStep(time_step)
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(orientation)
model.setMaxGridPoints(max_grid_points)
model.setInitialOxideThickness(seed_thickness)

use_gpu = cfg["useGpu"].lower()
if use_gpu == "gpu":
    model.setGpuMode(vps.GpuMode.Gpu)
elif use_gpu == "cpu":
    model.setGpuMode(vps.GpuMode.Cpu)
if cfg["gpuPreconditioner"].lower() == "ilu0":
    model.setGpuPreconditioner(vps.GpuPreconditioner.ILU0)

model.saveSurfaceMesh(domain, output_prefix + "_initial.vtp")

vps.Process(domain, model, 0.0).apply()

model.saveSurfaceMesh(domain, output_prefix + "_after.vtp")
model.saveVolumeMesh(domain, output_prefix + "_after")

print(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{model.estimatePlanarOxideThickness(seed_thickness):.4f} µm oxide."
)
print(
    f"Wrote {output_prefix}_initial.vtp, {output_prefix}_after.vtp, "
    f"and {output_prefix}_after_volume.vtu"
)
