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
import viennaps as ps

ps.setDimension(2)

# ── Config file ───────────────────────────────────────────────────────────────
config_file = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
cfg = ps.readConfigFile(config_file)

ps.setNumThreads(int(cfg["numThreads"]))
ps.Logger.setLogLevel(ps.LogLevel.INFO)

grid_delta = cfg["gridDelta"]
x_extent = cfg["xExtent"]
y_min = cfg["yMin"]
y_max = cfg["yMax"]
pad_oxide_thickness = cfg["padOxideThickness"]
mask_thickness = cfg["maskThickness"]
mask_edge = cfg["maskEdge"]
oxidation_time = cfg["oxidationTime"]
time_step = cfg["timeStep"]
temperature = cfg["temperature"]
pressure = cfg["pressure"]
oxidant = cfg["oxidant"]
orientation = str(int(cfg["orientation"]))
output_prefix = cfg["outputPrefix"]

# ── Build geometry ────────────────────────────────────────────────────────────

domain = ps.Domain(
    gridDelta=cfg["gridDelta"],
    xExtent=2 * cfg["xExtent"],
    boundary=ps.BoundaryType.REFLECTIVE_BOUNDARY,
)

# Si substrate flat at y = 0, then pad SiO2 at y = padOxideThickness.
ps.MakePlane(domain, 0.0, ps.Material.Si).apply()
ps.MakePlane(domain, pad_oxide_thickness, ps.Material.SiO2, True).apply()

# Si3N4 mask: box covering x ∈ [−xExtent, maskEdge], sitting on the pad oxide.
# No built-in ViennaPS helper for a half-mask, so construct it directly.
# The tiny contact epsilon places the mask bottom numerically inside the oxide
# so Cartesian stencils unambiguously see the mask/oxide boundary.
if mask_thickness > 0.0:
    mask_contact_eps = 1.0e-6
    mask_ls = ps.LevelSet(domain.getGrid())
    mask_geom = ps.MakeGeometry(
        mask_ls,
        ps.Box(
            [-x_extent, pad_oxide_thickness - mask_contact_eps],
            [mask_edge, pad_oxide_thickness + mask_thickness],
        ),
    )
    mask_geom.setIgnoreBoundaryConditions([False, True, False])
    mask_geom.apply()
    domain.insertNextLevelSetAsMaterial(mask_ls, ps.Material.Si3N4, False)

# ── Oxidation model ───────────────────────────────────────────────────────────
model = ps.Oxidation()
model.setTemperature(temperature)
model.setOxidant(oxidant)
model.setPressure(pressure)
model.setOrientation(orientation)
model.setTimeStep(time_step)
if "maxGridPoints" in cfg:
    model.setMaxGridPoints(cfg["maxGridPoints"])
model.setMechanicsIterations(int(cfg["mechanicsIterations"]))
model.setPressureIterations(int(cfg["pressureIterations"]))
model.setStokesIterations(int(cfg["stokesIterations"]))
model.setCouplingIterations(int(cfg["couplingIterations"]))
model.setCouplingTolerance(cfg["couplingTolerance"])
model.setMaskCouplingIterations(int(cfg["maskCouplingIterations"]))
model.setMaskCouplingTolerance(cfg["maskCouplingTolerance"])
mask_params = ps.ls.OxidationPresets.siliconNitrideMask1000C()
mask_params.referenceViscosity = cfg["maskReferenceViscosity"]
mask_params.poissonRatio = cfg["maskPoissonRatio"]
mask_params.youngModulus = cfg["maskYoungModulus"]
mask_params.unilateralContact = (
    True if cfg["maskUnilateralContact"] == "true" else False
)
mask_params.contactLoadRelaxation = cfg["maskContactLoadRelaxation"]
mask_params.contactReleaseFraction = cfg["maskContactReleaseFraction"]
_mode_str = cfg["maskContactMode"].lower().replace("-", "").replace("_", "")
mask_params.contactMode = (
    0
    if _mode_str in ("0", "kinematic")
    else (
        2
        if _mode_str
        in (
            "2",
            "3",
            "4",
            "elastic",
            "twoway",
            "feedback",
            "twowayelastic",
            "elasticfeedback",
        )
        else 1
    )  # default: oneway (aliases: "1", "oneway", "traction")
)
mask_params.maxIterations = int(cfg["maskTractionIterations"])
mask_params.tolerance = cfg["maskTractionTolerance"]
mask_params.relaxation = cfg["maskTractionRelaxation"]
mask_params.multigridSmootherOmega = cfg["maskSmootherOmega"]
mask_params.anchorBoundaryDirection = int(cfg["maskAnchorBoundaryDirection"])
mask_params.anchorBoundarySide = int(cfg["maskAnchorBoundarySide"])
mask_params.anchorBoundaryLayers = int(cfg["maskAnchorBoundaryLayers"])
model.setMaskParameters(mask_params)
model.setMechanicsTolerance(cfg["mechanicsTolerance"])
model.setPressureTolerance(cfg["pressureTolerance"])
model.setStokesTolerance(cfg["stokesTolerance"])

model.setGpuMode(cfg["useGpu"])
model.setGpuPreconditioner(cfg.get("gpuPreconditioner", "ilu0"))

model.saveSurfaceMesh(domain, f"{output_prefix}_step_000.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_step_000")

est = model.estimatePlanarOxideThickness(pad_oxide_thickness)
print(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{est:.4f} µm total oxide thickness."
)

# ── Time-stepping loop ────────────────────────────────────────────────────────
elapsed = 0.0
step = 0
time_eps = 1.0e-9 * oxidation_time

while oxidation_time - elapsed > time_eps:
    dt = min(time_step, oxidation_time - elapsed)
    if dt <= 0.0:
        break

    model.setTime(dt)
    model.setTimeStep(dt)
    ps.Process(domain, model, 0.0).apply()

    elapsed += dt
    step += 1

    fname = f"{output_prefix}_step_{step:03d}"
    model.saveSurfaceMesh(domain, fname + ".vtp")
    model.saveVolumeMesh(domain, fname)
    print(f"Wrote {fname} at t = {elapsed:.4f} hr.")

# ── Final output ───────────────────────────────────────────────────────────────
model.saveSurfaceMesh(domain, f"{output_prefix}_after.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_after")

print(
    f"Wrote {output_prefix}_after.vtp and {output_prefix}_after_volume.vtu "
    f"({step} time steps, elapsed = {elapsed:.4f} hr)"
)
