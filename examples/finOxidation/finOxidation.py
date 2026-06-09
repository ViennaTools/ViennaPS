#!/usr/bin/env python3
"""
Fin Structure Oxidation Example (ViennaPS)
==========================================
Simulates thermal oxidation of a rectangular Si fin on a flat Si substrate.
Oxide grows on all exposed surfaces: fin top, sidewalls, and the surrounding
flat substrate.

Geometry (2-D cross-section, extruded symmetrically in Z for 3-D):
    · Flat Si substrate at y = 0
    · Si fin centered at x = 0: width finWidth, height finHeight

Usage:
    python finOxidation.py [config.txt]

All lengths are in micrometers, time in hours, pressure in atm.

Coordinate convention:
    dim 0 = X  — cross-section direction (REFLECTIVE boundary, fin at x=0)
    dim 1 = Y  — height / growth direction (INFINITE boundary)
"""

import sys
import viennals
import viennals.d2 as ls
import viennaps as vps

viennals.setDimension(2)
vps.setDimension(2)

# ── Default parameters (match finOxidation/config.txt) ──────────────────────
cfg = {
    "numThreads":    16,
    "gridDelta":     0.05,
    "xExtent":       0.6,
    "yMin":         -1.0,
    "yMax":          2.0,
    "finWidth":      0.2,
    "finHeight":     0.5,
    "oxideThickness": 0.0,
    "oxidationTime": 0.2,
    "timeStep":      0.025,
    "temperature": 1000.0,
    "pressure":      1.0,
    "oxidant":     "wet",
    "orientation": "100",
    "outputPrefix": "ps_fin_oxidation",
    "maxGridPoints": 0,  # 0 = unlimited; set >0 to cap the Cartesian solve grid
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


# ── Config file ───────────────────────────────────────────────────────────────
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
_parse_config(config_file)

viennals.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.ERROR)

grid_delta      = cfg["gridDelta"]
x_extent        = cfg["xExtent"]
y_min           = cfg["yMin"]
y_max           = cfg["yMax"]
fin_width       = cfg["finWidth"]
fin_height      = cfg["finHeight"]
oxide_thickness = cfg["oxideThickness"]
oxidation_time  = cfg["oxidationTime"]
time_step       = cfg["timeStep"]
temperature     = cfg["temperature"]
pressure        = cfg["pressure"]
oxidant         = _parse_oxidant(cfg["oxidant"])
orientation     = _parse_orientation(cfg["orientation"])
output_prefix   = cfg["outputPrefix"]

# ── Domain bounds and boundary conditions ─────────────────────────────────────
BC     = viennals.BoundaryConditionEnum
bounds = [-x_extent, x_extent, y_min, y_max]
bcs    = [BC.REFLECTIVE_BOUNDARY, BC.INFINITE_BOUNDARY]

# ── Build Si fin level set ────────────────────────────────────────────────────
# Base: flat substrate at y = 0 (solid below).
si_ls = ls.Domain(bounds, bcs, grid_delta)
ls.MakeGeometry(si_ls, ls.Plane([0.0, 0.0], [0.0, 1.0])).apply()

# Fin box: centered at x = 0, from y ≈ 0 to y = finHeight.
# A tiny negative Y offset fuses the fin base seamlessly into the substrate.
fin_box = ls.Domain(bounds, bcs, grid_delta)
fin_geom = ls.MakeGeometry(
    fin_box,
    ls.Box([-fin_width / 2.0, -1e-6], [fin_width / 2.0, fin_height]),
)
fin_geom.setIgnoreBoundaryConditions([False, True])  # ignore INFINITE y
fin_geom.apply()
ls.BooleanOperation(si_ls, fin_box, viennals.BooleanOperationEnum.UNION).apply()

# ── Assemble ViennaPS domain ──────────────────────────────────────────────────
domain = vps.Domain()
domain.insertNextLevelSetAsMaterial(si_ls, vps.Material.Si, False)

# The deformation solver needs at least gridDelta of oxide between the two
# interfaces so that Cartesian solve nodes exist. Clamp the seed upward.
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
model.setInitialOxideThickness(seed_thickness)

if cfg["maxGridPoints"] > 0:
    model.setMaxGridPoints(int(cfg["maxGridPoints"]))

model.saveSurfaceMesh(domain, output_prefix + "_initial.vtp")
model.saveVolumeMesh(domain, output_prefix + "_initial")

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
