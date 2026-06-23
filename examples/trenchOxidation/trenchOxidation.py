#!/usr/bin/env python3
"""
Trench Structure Oxidation Example (ViennaPS)
=============================================
Simulates thermal oxidation of a rectangular Si trench etched into a flat
Si substrate. Oxide grows on all exposed Si surfaces: trench sidewalls,
trench floor, and the surrounding flat substrate.

Geometry (2-D cross-section, extruded symmetrically in Z for 3-D):
    · Flat Si substrate at y = 0
    · Trench centered at x = 0: width trenchWidth, depth trenchDepth (into Si)

Usage:
    python trenchOxidation.py [config.txt]

All lengths are in micrometers, time in hours, pressure in atm.
"""

import sys
import time
import viennaps as vps

# ── Default parameters (match trenchOxidation/config.txt) ───────────────────
cfg = {
    "dimensions":    2,
    "numThreads":    16,
    "gridDelta":     0.05,
    "xExtent":       0.6,
    "zExtent":       0.0,   # 3D only: half-depth in Z; defaults to xExtent if 0
    "trenchWidth":   0.3,
    "trenchDepth":   0.5,
    "oxideThickness": 0.0,
    "oxidationTime": 0.2,
    "temperature": 1000.0,
    "pressure":      1.0,
    "oxidant":            "wet",
    "orientation":        "100",
    "outputPrefix":       "ps_trench_oxidation",
    "maxGridPoints":      0,   # 0 = unlimited; set >0 to cap the Cartesian solve grid
    "useGpu":             "auto",   # auto | gpu | cpu
    "gpuPreconditioner":  "jacobi", # jacobi | ilu0
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
    if s in ("110", "<110>", "si110"):
        return vps.SiliconOrientation.Si110
    if s in ("111", "<111>", "si111"):
        return vps.SiliconOrientation.Si111
    if s in ("poly", "polysi"):
        return vps.SiliconOrientation.PolySi
    raise ValueError(f"Unknown orientation '{s}'. Use 100, 110, 111, or poly.")


# ── Config file ───────────────────────────────────────────────────────────────
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
_parse_config(config_file)

vps.setDimension(cfg["dimensions"])
vps.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

grid_delta      = cfg["gridDelta"]
x_extent        = cfg["xExtent"]
z_extent        = cfg["zExtent"] if cfg["zExtent"] > 0.0 else cfg["xExtent"]
trench_width    = cfg["trenchWidth"]
trench_depth    = cfg["trenchDepth"]
oxide_thickness = cfg["oxideThickness"]
oxidation_time  = cfg["oxidationTime"]
temperature     = cfg["temperature"]
pressure        = cfg["pressure"]
oxidant         = _parse_oxidant(cfg["oxidant"])
orientation     = _parse_orientation(cfg["orientation"])
output_prefix   = cfg["outputPrefix"]

# ── Build Si trench geometry ──────────────────────────────────────────────────
# MakeTrench's second constructor sets up the domain (REFLECTIVE X, INFINITE Y)
# and creates the trench: flat substrate at y=0 with a rectangular slot of
# width trenchWidth and depth trenchDepth centered at x=0.
domain = vps.Domain()
y_extent = 2.0 * z_extent if cfg["dimensions"] == 3 else 0.0
vps.MakeTrench(domain, gridDelta=grid_delta, xExtent=2.0 * x_extent, yExtent=y_extent,
               trenchWidth=trench_width, trenchDepth=trench_depth).apply()

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
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(orientation)
model.setInitialOxideThickness(seed_thickness)

use_gpu = cfg["useGpu"].lower()
if use_gpu == "gpu":
    model.setGpuMode(vps.GpuMode.Gpu)
elif use_gpu == "cpu":
    model.setGpuMode(vps.GpuMode.Cpu)
if cfg["gpuPreconditioner"].lower() == "ilu0":
    model.setGpuPreconditioner(vps.GpuPreconditioner.ILU0)

if cfg["maxGridPoints"] > 0:
    model.setMaxGridPoints(int(cfg["maxGridPoints"]))

model.saveSurfaceMesh(domain, output_prefix + "_initial.vtp")
model.saveVolumeMesh(domain, output_prefix + "_initial")

t0 = time.perf_counter()
vps.Process(domain, model, 0.0).apply()
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
