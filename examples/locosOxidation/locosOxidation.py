#!/usr/bin/env python3
"""
LOCOS (Local Oxidation of Silicon) Example (ViennaPS)
======================================================
Simulates the bird's-beak oxide profile that forms when a Si3N4 pad
mask constrains lateral oxidation at its edges.

Geometry (2-D cross-section):
    · Si substrate at y = 0
    · Pad SiO2 layer of thickness padOxideThickness grown on Si
    · Si3N4 mask box covering x < maskEdge, sitting on the pad oxide

The ViennaPS Oxidation model auto-detects Si3N4 and activates LOCOS
physics: mask-bending + constrained-ambient advection. The example saves a
surface mesh every timeStep hours; the model may use smaller CFL-limited
internal physics steps between saved meshes.

Usage:
    python locosOxidation.py [config.txt]

All lengths are in micrometers, time in hours, pressure in atm.
"""

import sys
import os
import viennals as vls
import viennaps as vps

vps.setDimension(2)

# ── Default parameters (match locosOxidation/config.txt) ─────────────────────
cfg = {
    "numThreads":          16,
    "gridDelta":           0.05,
    "xExtent":             4.0,
    "yMin":               -1.0,
    "yMax":                2.0,
    "padOxideThickness":   0.05,
    "maskThickness":       0.3,
    "maskEdge":            0.0,
    "oxidationTime":       1.0,
    "timeStep":            0.1,
    "temperature":      1000.0,
    "pressure":            1.0,
    "oxidant":           "wet",
    "orientation":       "100",
    "maxGridPoints":          5000000,
    "outputPrefix":           "ps_locos",
    "mechanicsIterations":    200,
    "mechanicsTolerance":     1e-4,
    "pressureIterations":     500,
    "pressureTolerance":      1e-6,
    "stokesIterations":       200,
    "stokesTolerance":        1e-7,
    "couplingIterations":     8,
    "couplingTolerance":      1e-4,
    "maskCouplingIterations": 8,
    "maskCouplingTolerance":  0.02,
    "maskReferenceViscosity": 5.0e11,
    "maskPoissonRatio":       0.27,
    "maskContactMode":           "oneway",
    "maskYoungModulus":          270e9,
    "maskUnilateralContact":     False,
    "maskContactLoadRelaxation": 1.0,
    "maskContactReleaseFraction": 1e-2,
    "maskTractionIterations":    10000,
    "maskTractionTolerance":     1.0e-5,
    "maskTractionRelaxation":    0.9,
    "maskSmootherOmega":         1.0,
    "maskAnchorBoundaryDirection": 0,
    "maskAnchorBoundarySide":   -1,
    "maskAnchorBoundaryLayers":  1,
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
                val = line[eq + 1:].split("#")[0].strip()  # strip inline comments
                if key in cfg:
                    t = type(cfg[key])
                    if t is bool:
                        cfg[key] = val.lower() in ("true", "1", "yes")
                    else:
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

vps.setNumThreads(cfg["numThreads"])
vps.Logger.setLogLevel(vps.LogLevel.INFO)

grid_delta          = cfg["gridDelta"]
x_extent            = cfg["xExtent"]
y_min               = cfg["yMin"]
y_max               = cfg["yMax"]
pad_oxide_thickness = cfg["padOxideThickness"]
mask_thickness      = cfg["maskThickness"]
mask_edge           = cfg["maskEdge"]
oxidation_time      = cfg["oxidationTime"]
time_step           = cfg["timeStep"]
temperature         = cfg["temperature"]
pressure            = cfg["pressure"]
oxidant             = _parse_oxidant(cfg["oxidant"])
orientation         = _parse_orientation(cfg["orientation"])
max_grid_points     = cfg["maxGridPoints"]
output_prefix       = cfg["outputPrefix"]

# Tiny offset so Cartesian stencils unambiguously see the mask/oxide interface.
mask_contact_eps = 1.0e-6

# ── Domain bounds and boundary conditions ────────────────────────────────────
BC = vps.BoundaryType
bounds = [-x_extent, x_extent, y_min, y_max]
bcs    = [BC.REFLECTIVE_BOUNDARY, BC.INFINITE_BOUNDARY]

# ── Build level sets ──────────────────────────────────────────────────────────

# Si substrate: flat plane at y = 0.
si_ls = vls.Domain(bounds, bcs, grid_delta)
vls.MakeGeometry(si_ls, vls.Plane([0.0, 0.0], [0.0, 1.0])).apply()

# Pad SiO2: geometrically advance Si surface by padOxideThickness.
oxide_ls = vls.Domain(si_ls)
vls.GeometricAdvect(
    oxide_ls, vls.SphereDistribution(pad_oxide_thickness)
).apply()

# Si3N4 mask: rectangular box covering x ∈ [-xExtent, maskEdge],
# y ∈ [pad_oxide_top - eps, pad_oxide_top + maskThickness].
pad_oxide_top = pad_oxide_thickness
mask_ls = vls.Domain(bounds, bcs, grid_delta)
mask_geom = vls.MakeGeometry(
    mask_ls,
    vls.Box(
        [-x_extent, pad_oxide_top - mask_contact_eps],
        [mask_edge,  pad_oxide_top + mask_thickness],
    ),
)
mask_geom.setIgnoreBoundaryConditions([False, True, False])  # ignore INFINITE y boundary
mask_geom.apply()

# ── Assemble ViennaPS domain ──────────────────────────────────────────────────
domain = vps.Domain()
domain.insertNextLevelSetAsMaterial(si_ls,    vps.Material.Si,    False)
domain.insertNextLevelSetAsMaterial(oxide_ls, vps.Material.SiO2,  False)

# Only add mask if thickness is positive. Setting maskThickness <= 0 disables
# LOCOS physics and uses standard oxidation instead.
if mask_thickness > 0.0:
    domain.insertNextLevelSetAsMaterial(mask_ls,  vps.Material.Si3N4, False)

# ── Oxidation model ───────────────────────────────────────────────────────────
model = vps.Oxidation()
model.setTemperature(temperature)
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(orientation)
model.setTimeStep(time_step)
model.setMaxGridPoints(max_grid_points)
model.setMechanicsIterations(cfg["mechanicsIterations"])
model.setPressureIterations(cfg["pressureIterations"])
model.setStokesIterations(cfg["stokesIterations"])
model.setCouplingIterations(cfg["couplingIterations"])
model.setCouplingTolerance(cfg["couplingTolerance"])
model.setMaskCouplingIterations(cfg["maskCouplingIterations"])
model.setMaskCouplingTolerance(cfg["maskCouplingTolerance"])
mask_params = vls.OxidationPresets.siliconNitrideMask1000C()
mask_params.referenceViscosity = cfg["maskReferenceViscosity"]
mask_params.poissonRatio = cfg["maskPoissonRatio"]
mask_params.youngModulus = cfg["maskYoungModulus"]
mask_params.unilateralContact = cfg["maskUnilateralContact"]
mask_params.contactLoadRelaxation = cfg["maskContactLoadRelaxation"]
mask_params.contactReleaseFraction = cfg["maskContactReleaseFraction"]
_mode_str = cfg["maskContactMode"].lower().replace("-", "").replace("_", "")
mask_params.contactMode = (
    0 if _mode_str in ("0", "kinematic") else
    2 if _mode_str in ("2", "3", "4", "elastic", "twoway", "feedback",
                       "twowayelastic", "elasticfeedback") else
    1  # default: oneway (aliases: "1", "oneway", "traction")
)
mask_params.maxIterations = int(cfg["maskTractionIterations"])
mask_params.tolerance = cfg["maskTractionTolerance"]
mask_params.relaxation = cfg["maskTractionRelaxation"]
mask_params.multigridSmootherOmega = cfg["maskSmootherOmega"]
mask_params.anchorBoundaryDirection = cfg["maskAnchorBoundaryDirection"]
mask_params.anchorBoundarySide = cfg["maskAnchorBoundarySide"]
mask_params.anchorBoundaryLayers = cfg["maskAnchorBoundaryLayers"]
model.setMaskParameters(mask_params)
model.setMechanicsTolerance(cfg["mechanicsTolerance"])
model.setPressureTolerance(cfg["pressureTolerance"])
model.setStokesTolerance(cfg["stokesTolerance"])

model.saveSurfaceMesh(domain, f"{output_prefix}_step_000.vtp")

est = model.estimatePlanarOxideThickness(pad_oxide_thickness)
print(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{est:.4f} µm total oxide thickness."
)

# ── Time-stepping loop ────────────────────────────────────────────────────────
elapsed = 0.0
step    = 0
time_eps = 1.0e-9 * oxidation_time

while oxidation_time - elapsed > time_eps:
    dt = min(time_step, oxidation_time - elapsed)
    if dt <= 0.0:
        break

    model.setTime(dt)
    model.setTimeStep(dt)
    vps.Process(domain, model, 0.0).apply()

    elapsed += dt
    step    += 1

    fname = f"{output_prefix}_step_{step:03d}.vtp"
    model.saveSurfaceMesh(domain, fname)
    print(f"Wrote {fname} at t = {elapsed:.4f} hr.")

# ── Final output ───────────────────────────────────────────────────────────────
model.saveSurfaceMesh(domain, f"{output_prefix}_after.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_after")

print(f"Wrote {output_prefix}_after.vtp and {output_prefix}_after_volume.vtu "
      f"({step} time steps, elapsed = {elapsed:.4f} hr)")
