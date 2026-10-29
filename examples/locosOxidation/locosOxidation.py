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
physics: the nitride joins the oxide mechanics solve as an elastic body
that creeps. The example saves a surface mesh every timeStep hours; the
model may use smaller CFL-limited internal physics steps between saved
meshes.

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
ps.Logger.setLogLevel(cfg.get("logLevel", "info"))
log = ps.Logger.getInstance()

# Keys this example reads; anything else is a typo or a retired key.
KNOWN_KEYS = {
    "numThreads", "logLevel", "useGpu", "gridDelta", "xExtent", "yMin", "yMax",
    "padOxideThickness", "maskThickness", "maskEdge", "maskPermeabilityRatio",
    "maskBarrierOnly", "dealGroveB", "dealGroveBoA", "oxidationTime",
    "initialTimeStep", "regridDelta", "stressDependentOxidation",
    "reactionActivationVolume", "diffusionActivationVolume",
    "shearActivationVolume", "timeStep", "temperature", "pressure", "oxidant",
    "orientation", "maxGridPoints", "outputPrefix", "mechanicsIterations",
    "mechanicsTolerance", "couplingIterations", "couplingTolerance",
    "maskPoissonRatio", "maskYoungModulus", "maskViscosity",
    "maskShearActivationVolume",
}
for key in cfg.keys() - KNOWN_KEYS:
    log.addWarning(f"Unknown config key '{key}' ignored.").print()


def flag(key, default):
    return str(cfg.get(key, default)).lower() in ("true", "yes", "on", "1.0")


grid_delta = cfg["gridDelta"]
x_extent = cfg["xExtent"]
pad_oxide_thickness = cfg["padOxideThickness"]
mask_thickness = cfg["maskThickness"]
mask_edge = cfg["maskEdge"]
oxidation_time = cfg["oxidationTime"]
time_step = cfg["timeStep"]
temperature = cfg["temperature"]
orientation = cfg["orientation"]
if isinstance(orientation, float):  # "100" is read as a number
    orientation = str(int(orientation))
output_prefix = cfg["outputPrefix"]

# ── Build geometry ────────────────────────────────────────────────────────────

# Asymmetric y-bounds: yMin below Si surface, yMax above anticipated oxide height.
bounds = [-x_extent, x_extent, cfg["yMin"], cfg["yMax"]]
bcs = [ps.BoundaryType.REFLECTIVE_BOUNDARY, ps.BoundaryType.INFINITE_BOUNDARY]
domain = ps.Domain(bounds, bcs, grid_delta)

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
            # Overhang the left domain edge: a box ending exactly on the
            # boundary leaves uncovered oxide cells there, which classify as
            # AMBIENT and admit unbarriered oxidant.  Same applies on the
            # right for a full-width mask (set maskEdge > xExtent for that).
            [-2.0 * x_extent, pad_oxide_thickness - mask_contact_eps],
            [mask_edge, pad_oxide_thickness + mask_thickness],
        ),
    )
    mask_geom.setIgnoreBoundaryConditions([False, True, False])
    mask_geom.apply()
    domain.insertNextLevelSetAsMaterial(mask_ls, ps.Material.Si3N4, False)

# ── Oxidation model ───────────────────────────────────────────────────────────
model = ps.Oxidation()
model.setTemperature(temperature)
model.setOxidant(cfg["oxidant"])
model.setPressure(cfg["pressure"])
model.setOrientation(orientation)
# Diffusion through the mask: gamma = D_SiO2/D_mask (0 = impermeable mask).
if cfg.get("maskPermeabilityRatio", 0.0) > 0.0:
    model.setMaskPermeabilityRatio(cfg["maskPermeabilityRatio"])
# Mask as diffusion barrier only (mechanically transparent).
model.setMaskBarrierOnly(flag("maskBarrierOnly", False))
# Deal-Grove overrides (0 = built-in Arrhenius table).
if cfg.get("dealGroveB", 0.0) > 0.0 or cfg.get("dealGroveBoA", 0.0) > 0.0:
    model.setDealGroveRates(cfg.get("dealGroveB", 0.0), cfg.get("dealGroveBoA", 0.0))
model.setTimeStep(time_step)
if "initialTimeStep" in cfg:  # first-substep cap; 0 = model default
    model.setInitialTimeStep(cfg["initialTimeStep"])
# One-shot regrid to this delta once the thinnest oxide exceeds 2x it.
if cfg.get("regridDelta", 0.0) > 0.0:
    model.setRegrid(cfg["regridDelta"])
# Stress-dependent kinetics (on by default, Kao-calibrated).
model.setStressDependentOxidation(flag("stressDependentOxidation", True))
# Activation volumes (m^3); 0 = model defaults.
if cfg.get("reactionActivationVolume", 0.0) > 0.0:
    model.setReactionActivationVolume(cfg["reactionActivationVolume"])
if cfg.get("diffusionActivationVolume", 0.0) > 0.0:
    model.setDiffusionActivationVolume(cfg["diffusionActivationVolume"])
# Eyring oxide viscosity (m^3); < 0 = model default (300 A^3), 0 = linear.
if cfg.get("shearActivationVolume", -1.0) >= 0.0:
    model.setShearActivationVolume(cfg["shearActivationVolume"])
if "maxGridPoints" in cfg:
    model.setMaxGridPoints(int(cfg["maxGridPoints"]))
model.setMechanicsIterations(int(cfg["mechanicsIterations"]))
model.setMechanicsTolerance(cfg["mechanicsTolerance"])
model.setCouplingIterations(int(cfg["couplingIterations"]))
model.setCouplingTolerance(cfg["couplingTolerance"])
# Nitride: elastic with Maxwell creep; maskViscosity (Pa hr) and
# maskShearActivationVolume (m^3) < 0 take the Si3N4 preset at the process
# temperature (Senez 1996), 0 = linear-elastic / linear viscosity.
mask_params = ps.ls.OxidationPresets.siliconNitrideMask(temperature)
mask_params.poissonRatio = cfg["maskPoissonRatio"]
mask_params.youngModulus = cfg["maskYoungModulus"]
if cfg.get("maskViscosity", -1.0) >= 0.0:
    mask_params.viscosity = cfg["maskViscosity"]
if cfg.get("maskShearActivationVolume", -1.0) >= 0.0:
    mask_params.shearActivationVolume = cfg["maskShearActivationVolume"]
model.setMaskParameters(mask_params)
model.setGpuMode(cfg.get("useGpu", "cpu"))

model.saveSurfaceMesh(domain, f"{output_prefix}_step_000.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_step_000")

model.setTime(oxidation_time)
est = model.estimatePlanarOxideThickness(pad_oxide_thickness)
log.addInfo(
    f"Planar Deal-Grove estimate for {oxidation_time} hr at {temperature} °C: "
    f"{est:.4f} µm total oxide thickness."
).print()

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
    log.addInfo(f"Wrote {fname} at t = {elapsed:.4f} hr.").print()

# ── Final output ───────────────────────────────────────────────────────────────
model.saveSurfaceMesh(domain, f"{output_prefix}_after.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_after")

log.addInfo(
    f"Wrote {output_prefix}_after.vtp and {output_prefix}_after_volume.vtu "
    f"({step} time steps, elapsed = {elapsed:.4f} hr)"
).print()
