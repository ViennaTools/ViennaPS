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
The timeStep setting is a maximum internal oxidation step; CFL-limited
subcycling is automatic.

Coordinate convention:
    dim 0 = X  — lateral step direction (REFLECTIVE boundary)
    dim 1 = Y  — height / growth direction (INFINITE boundary)
"""

import sys
import viennals
import viennals.d2 as ls
import viennaps as vps

viennals.setDimension(2)
vps.setDimension(2)

# ── Default parameters (match stepOxidation/config.txt) ─────────────────────
cfg = {
    "numThreads":      16,
    "gridDelta":       0.05,
    "xExtent":         1.0,
    "yMin":           -2.0,
    "yMax":            4.0,
    "stepX":           0.0,
#    "stepWidth":       2.0,
    "leftSiTop":       0.0,
    "rightSiTop":      1.0,
    "oxideThickness":  0.0,
    "oxidationTime":   0.05,
    "timeStep":        0.01,
    "temperature":  1000.0,
    "pressure":        1.0,
    "oxidant":       "wet",
    "orientation":   "100",
    "maxGridPoints": 5000000,
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

viennals.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

grid_delta      = cfg["gridDelta"]
x_extent        = cfg["xExtent"]
y_min           = cfg["yMin"]
y_max           = cfg["yMax"]
step_x          = cfg["stepX"]
#step_width      = cfg["stepWidth"]
left_si_top     = cfg["leftSiTop"]
right_si_top    = cfg["rightSiTop"]
oxide_thickness = cfg["oxideThickness"]
oxidation_time  = cfg["oxidationTime"]
time_step       = cfg["timeStep"]
temperature     = cfg["temperature"]
pressure        = cfg["pressure"]
oxidant         = _parse_oxidant(cfg["oxidant"])
orientation     = _parse_orientation(cfg["orientation"])
max_grid_points = cfg["maxGridPoints"]
output_prefix   = cfg["outputPrefix"]

# ── Domain bounds and boundary conditions ────────────────────────────────────
BC = viennals.BoundaryConditionEnum
bounds = [-x_extent, x_extent, y_min, y_max]
bcs    = [BC.REFLECTIVE_BOUNDARY, BC.INFINITE_BOUNDARY]

# ── Build Si step level set ──────────────────────────────────────────────────
lo_top = min(left_si_top, right_si_top)
hi_top = max(left_si_top, right_si_top)

# Base: horizontal plane at y = lo_top (the lower surface).
si_ls = ls.Domain(bounds, bcs, grid_delta)
ls.MakeGeometry(si_ls, ls.Plane([0.0, lo_top], [0.0, 1.0])).apply()

if abs(hi_top - lo_top) > 1e-10:
    # Raised block: extend 3× beyond the domain boundary so the reflective BC
    # sees solid Si interior rather than a box face at x=±x_extent.  This
    # avoids a spurious vertical Si wall at the reflective boundary that would
    # block oxide growth on the top surface near the edge.
    large = 3.0 * x_extent
    raised_block = ls.Domain(bounds, bcs, grid_delta)
    if left_si_top > right_si_top:
        # Raised platform is on the LEFT side
        box = ls.Box([-large, lo_top], [step_x, hi_top])
    else:
        # Raised platform is on the RIGHT side
        box = ls.Box([step_x, lo_top], [large, hi_top])
    geom = ls.MakeGeometry(raised_block, box)
    geom.setIgnoreBoundaryConditions([True, True])
    geom.apply()
    ls.BooleanOperation(
        si_ls, raised_block, viennals.BooleanOperationEnum.UNION
    ).apply()

# ── Assemble ViennaPS domain ─────────────────────────────────────────────────
domain = vps.Domain()
domain.insertNextLevelSetAsMaterial(si_ls, vps.Material.Si, False)

# The deformation solver needs the oxide to be at least gridDelta thick so
# that Cartesian solve nodes exist between the two surfaces.  Clamp the seed
# upward when the user specifies a sub-grid or zero initial oxide.
seed_thickness = max(oxide_thickness, grid_delta)
ambient_ls = ls.Domain(si_ls)
ls.GeometricAdvect(ambient_ls, ls.SphereDistribution(seed_thickness)).apply()
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
