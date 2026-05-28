#!/usr/bin/env python3
"""
Step Oxidation Example (ViennaPS)
===================================
Simulates thermal oxidation of a silicon step geometry using the
Deal-Grove + Stokes-flow deformation model.

Usage:
    python stepOxidation.py [config.txt]

If a config file is given its key=value pairs override the defaults below.
All lengths are in micrometers, time in hours, pressure in atm.

Coordinate convention:
    dim 0 = X  — lateral step direction (REFLECTIVE boundary)
    dim 1 = Y  — height / growth direction (INFINITE boundary)
"""

import sys
import viennals
import viennals.d2 as ls
import viennaps2d as vps

# ── Default parameters (match stepOxidation/config.txt) ─────────────────────
cfg = {
    "numThreads":      16,
    "gridDelta":       0.05,
    "xExtent":         4.0,
    "yMin":           -2.0,
    "yMax":            4.0,
    "stepX":           0.0,
    "leftSiTop":       0.0,
    "rightSiTop":      1.0,
    "oxideThickness":  0.0,
    "oxidationTime":   0.05,
    "timeStep":        0.01,
    "temperature":  1000.0,
    "pressure":        1.0,
    "oxidant":       "wet",
    "orientation":   "100",
    "outputPrefix":  "ps_step_oxidation",
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
                key, val = line[:eq].strip(), line[eq + 1:].strip()
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

viennals.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

grid_delta      = cfg["gridDelta"]
x_extent        = cfg["xExtent"]
y_min           = cfg["yMin"]
y_max           = cfg["yMax"]
step_x          = cfg["stepX"]
left_si_top     = cfg["leftSiTop"]
right_si_top    = cfg["rightSiTop"]
oxide_thickness = cfg["oxideThickness"]
oxidation_time  = cfg["oxidationTime"]
time_step       = cfg["timeStep"]
temperature     = cfg["temperature"]
pressure        = cfg["pressure"]
oxidant         = _parse_oxidant(cfg["oxidant"])
orientation     = _parse_orientation(cfg["orientation"])
output_prefix   = cfg["outputPrefix"]

# ── Domain bounds and boundary conditions ────────────────────────────────────
BC = viennals.BoundaryConditionEnum
bounds = [-x_extent, x_extent, y_min, y_max]
bcs    = [BC.REFLECTIVE_BOUNDARY, BC.INFINITE_BOUNDARY]

# ── Build Si step level set ──────────────────────────────────────────────────
# Base: horizontal plane at y = left_si_top (everything below is solid Si).
si_ls = ls.Domain(bounds, bcs, grid_delta)
ls.MakeGeometry(si_ls, ls.Plane([0.0, left_si_top], [0.0, 1.0])).apply()

# Raised block at x > step_x, occupying y ∈ [left_si_top, right_si_top].
right_block = ls.Domain(bounds, bcs, grid_delta)
geom = ls.MakeGeometry(
    right_block,
    ls.Box([step_x, left_si_top], [x_extent, right_si_top]),
)
geom.setIgnoreBoundaryConditions([False, True, False])  # ignore INFINITE y boundary
geom.apply()
ls.BooleanOperation(
    si_ls, right_block, viennals.BooleanOperationEnum.UNION
).apply()

# ── Assemble ViennaPS domain ─────────────────────────────────────────────────
domain = vps.Domain()
domain.insertNextLevelSetAsMaterial(si_ls, vps.Material.Si, False)

if oxide_thickness > 0.0:
    ambient_ls = ls.Domain(si_ls)          # copy Si surface → grow oxide off it
    ls.GeometricAdvect(
        ambient_ls, ls.SphereDistribution(oxide_thickness)
    ).apply()
    domain.insertNextLevelSetAsMaterial(ambient_ls, vps.Material.SiO2, False)

domain.saveSurfaceMesh(output_prefix + "_initial.vtp")

# ── Oxidation model ───────────────────────────────────────────────────────────
model = vps.Oxidation()
model.setTemperature(temperature)
model.setTime(oxidation_time)
model.setTimeStep(time_step)
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(orientation)
model.setInitialOxideThickness(oxide_thickness)

vps.Process(domain, model, 0.0).apply()

domain.saveSurfaceMesh(output_prefix + "_after.vtp")

print(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{model.estimatePlanarOxideThickness(oxide_thickness):.4f} µm oxide."
)
print(f"Wrote {output_prefix}_initial.vtp and {output_prefix}_after.vtp")
