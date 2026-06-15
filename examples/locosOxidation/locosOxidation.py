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
import viennals as vls
import viennaps as vps

vps.setDimension(2)

# ── Default parameters (match locosOxidation/config.txt) ─────────────────────
cfg = {
    "numThreads":          4,
    "gridDelta":           0.05,   # coarse default; config.txt uses 0.005 (GPU)
    "xExtent":             1.0,
    "yMin":               -1.0,
    "yMax":                2.0,
    "padOxideThickness":   0.03,
    "maskThickness":       0.05,
    "maskEdge":            0.0,
    "oxidationTime":       0.1,
    "timeStep":            0.01,
    "temperature":      1000.0,
    "pressure":            1.0,
    "oxidant":           "wet",
    "orientation":       "100",
    "maxGridPoints":          5000000,
    "outputPrefix":           "ps_locos",
    # SIMPLE mechanics solver parameters (must be generous for convergence)
    "mechanicsIterations":    300,
    "mechanicsTolerance":     5e-3,
    "pressureIterations":     500,
    "pressureTolerance":      1e-3,
    "stokesIterations":       500,
    "stokesTolerance":        1e-3,
    "couplingIterations":     100,
    "couplingTolerance":      2e-2,
    "maskCouplingIterations": 30,
    "maskCouplingTolerance":  1e-2,
    "maskReferenceViscosity": 5.0e11,
    "maskPoissonRatio":       0.27,
    "maskContactMode":           "twoway",
    "maskYoungModulus":          270e9,
    "maskUnilateralContact":     True,
    "maskContactLoadRelaxation": 0.25,
    "maskContactReleaseFraction": 5e-3,
    "maskTractionIterations":    10000,
    "maskTractionTolerance":     1.0e-5,
    "maskTractionRelaxation":    0.9,
    "maskSmootherOmega":         1.0,
    "maskAnchorBoundaryDirection": 0,
    "maskAnchorBoundarySide":   -1,
    "maskAnchorBoundaryLayers":  1,
    "useGpu":            "cpu",    # cpu | gpu
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

# ── Build geometry ────────────────────────────────────────────────────────────

# Asymmetric y-bounds: yMin below Si surface, yMax above anticipated oxide height.
BC = vps.BoundaryType
bounds = [-x_extent, x_extent, y_min, y_max]
bcs    = [BC.REFLECTIVE_BOUNDARY, BC.INFINITE_BOUNDARY]

domain = vps.Domain(bounds, bcs, grid_delta)

# Si substrate flat at y = 0, then pad SiO2 at y = padOxideThickness.
vps.MakePlane(domain, 0., vps.Material.Si).apply()
vps.MakePlane(domain, pad_oxide_thickness, vps.Material.SiO2, True).apply()

# Si3N4 mask: box covering x ∈ [−xExtent, maskEdge], sitting on the pad oxide.
# No built-in ViennaPS helper for a half-mask, so construct it directly.
# The tiny contact epsilon places the mask bottom numerically inside the oxide
# so Cartesian stencils unambiguously see the mask/oxide boundary.
if mask_thickness > 0.0:
    mask_contact_eps = 1.0e-6
    mask_ls = vls.Domain(bounds, bcs, grid_delta)
    mask_geom = vls.MakeGeometry(
        mask_ls,
        vls.Box(
            [-x_extent, pad_oxide_thickness - mask_contact_eps],
            [mask_edge,  pad_oxide_thickness + mask_thickness],
        ),
    )
    mask_geom.setIgnoreBoundaryConditions([False, True, False])
    mask_geom.apply()
    domain.insertNextLevelSetAsMaterial(mask_ls, vps.Material.Si3N4, False)

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

use_gpu = cfg["useGpu"].lower()
if use_gpu == "gpu":
    model.setGpuMode(vps.GpuMode.Gpu)
elif use_gpu == "cpu":
    model.setGpuMode(vps.GpuMode.Cpu)
if cfg["gpuPreconditioner"].lower() == "ilu0":
    model.setGpuPreconditioner(vps.GpuPreconditioner.ILU0)

model.saveSurfaceMesh(domain, f"{output_prefix}_step_000.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_step_000")

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

    fname = f"{output_prefix}_step_{step:03d}"
    model.saveSurfaceMesh(domain, fname + ".vtp")
    model.saveVolumeMesh(domain, fname)
    print(f"Wrote {fname} at t = {elapsed:.4f} hr.")

# ── Final output ───────────────────────────────────────────────────────────────
model.saveSurfaceMesh(domain, f"{output_prefix}_after.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_after")

print(f"Wrote {output_prefix}_after.vtp and {output_prefix}_after_volume.vtu "
      f"({step} time steps, elapsed = {elapsed:.4f} hr)")
