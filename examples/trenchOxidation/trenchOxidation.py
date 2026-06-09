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

Note: the deformation solver needs at least 2×gridDelta of initial oxide to
converge on concave trench surfaces. The default oxideThickness in config.txt
(0.1 µm at gridDelta=0.05) satisfies this requirement; if you change gridDelta
keep oxideThickness ≥ 2×gridDelta.

Usage:
    python trenchOxidation.py [config.txt]

All lengths are in micrometers, time in hours, pressure in atm.

Coordinate convention:
    dim 0 = X  — cross-section direction (REFLECTIVE boundary, trench at x=0)
    dim 1 = Y  — height / growth direction (INFINITE boundary)
"""

import sys
import viennals
import viennals.d2 as ls
import viennaps as vps

viennals.setDimension(2)
vps.setDimension(2)

# ── Default parameters (match trenchOxidation/config.txt) ───────────────────
cfg = {
    "numThreads":    16,
    "gridDelta":     0.05,
    "xExtent":       0.6,
    "yMin":         -1.5,
    "yMax":          1.5,
    "trenchWidth":   0.3,
    "trenchDepth":   0.5,
    "oxideThickness": 0.1,   # ≥ 2×gridDelta required for deformation solver
    "oxidationTime": 0.2,
    "timeStep":      0.025,
    "temperature": 1000.0,
    "pressure":      1.0,
    "oxidant":     "wet",
    "orientation": "100",
    "outputPrefix": "ps_trench_oxidation",
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
trench_width    = cfg["trenchWidth"]
trench_depth    = cfg["trenchDepth"]
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

# ── Build Si trench level set ─────────────────────────────────────────────────
# Base: flat substrate at y = 0 (solid below).
si_ls = ls.Domain(bounds, bcs, grid_delta)
ls.MakeGeometry(si_ls, ls.Plane([0.0, 0.0], [0.0, 1.0])).apply()

# Trench void box: the cavity to subtract from the substrate.
# Extends slightly above y=0 and below y=-trenchDepth for a clean boolean cut.
void_box = ls.Domain(bounds, bcs, grid_delta)
void_geom = ls.MakeGeometry(
    void_box,
    ls.Box(
        [-trench_width / 2.0, -trench_depth - grid_delta],
        [ trench_width / 2.0,  grid_delta],
    ),
)
void_geom.setIgnoreBoundaryConditions([False, True])  # ignore INFINITE y
void_geom.apply()
ls.BooleanOperation(
    si_ls, void_box, viennals.BooleanOperationEnum.RELATIVE_COMPLEMENT
).apply()

# ── Assemble ViennaPS domain ──────────────────────────────────────────────────
domain = vps.Domain()
domain.insertNextLevelSetAsMaterial(si_ls, vps.Material.Si, False)

# For the trench geometry the deformation solver needs at least 2×gridDelta of
# oxide to converge on the concave walls. Use max(oxideThickness, gridDelta);
# the config.txt default (0.1 µm at gridDelta=0.05) already satisfies this.
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
