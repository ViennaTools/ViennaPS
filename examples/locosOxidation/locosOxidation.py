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
physics: the nitride joins the oxide mechanics solve as a linear-elastic
body. The example saves a surface mesh every timeStep hours; the model may
use smaller CFL-limited internal physics steps between saved meshes.

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
    # Diffusion through the mask: gamma = D_SiO2/D_mask.  The mask body is
    # meshed into the oxidant solve with D/gamma, so maskThickness is the
    # physical barrier thickness and must be resolved by gridDelta.
    # 0 = impermeable mask (default).
    "maskPermeabilityRatio": 0.0,
    # Mask as diffusion barrier only (mechanically transparent, carried by the
    # oxide).  The mode for full-wafer barrier validation.
    "maskBarrierOnly":     False,
    # Deal-Grove overrides (0 = built-in Arrhenius table).
    "dealGroveB":          0.0,
    "dealGroveBoA":        0.0,
    "oxidationTime":       0.1,
    "initialTimeStep":     0.0,   # first-substep cap; 0 = model default
    # One-shot persistent regrid: coarsen the whole stack to this delta (um)
    # once the thinnest oxide layer exceeds 2x it.  0 = disabled.
    "regridDelta":         0.0,
    # Stress-dependent oxidation kinetics (on by default, Kao-calibrated):
    # false = stress-free Deal-Grove kinetics.
    "stressDependentOxidation":  True,
    # Activation volumes (m^3) of the reaction (interface normal stress) and
    # the diffusivity (pressure); 0 = model defaults (2.5e-29, 4e-29).
    "reactionActivationVolume":  0.0,
    "diffusionActivationVolume": 0.0,
    # Stress-dependent oxide viscosity (Sutardja-Oldham): shear activation
    # volume (m^3); < 0 = model default (3e-28 = 300 A^3), 0 = linear.
    "shearActivationVolume":     -1.0,
    "timeStep":            0.01,
    "temperature":      1000.0,
    "pressure":            1.0,
    "oxidant":           "wet",
    "orientation":       "100",
    "maxGridPoints":          5000000,
    "outputPrefix":           "ps_locos",
    # Mechanics: viscosity iterations of the stress-dependent oxide (one
    # velocity-pressure solve each) until the relative velocity change is
    # below mechanicsTolerance.
    "mechanicsIterations":    300,
    "mechanicsTolerance":     1e-4,
    "couplingIterations":     100,
    "couplingTolerance":      2e-2,
    # Nitride mask: elastic body with Maxwell creep in the oxide mechanics
    # solve.  maskViscosity (Pa hr) / maskShearActivationVolume (m^3): < 0
    # takes the Si3N4 preset at the process temperature (Senez 1996), 0 = a
    # linear-elastic mask / a linear viscosity.
    "maskPoissonRatio":       0.27,
    "maskYoungModulus":          270e9,
    "maskViscosity":             -1.0,
    "maskShearActivationVolume": -1.0,
    "useGpu":            "cpu",    # cpu | gpu | auto
    "logLevel":          "info",   # debug | timing | intermediate | info | warning | error
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
                else:
                    print(f"Warning: unknown config key '{key}' ignored.",
                          file=sys.stderr)
    except FileNotFoundError:
        print(f"Warning: could not open {path}, using defaults.", file=sys.stderr)


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
vps.Logger.setLogLevel({
    "debug": vps.LogLevel.DEBUG,
    "timing": vps.LogLevel.TIMING,
    "intermediate": vps.LogLevel.INTERMEDIATE,
    "info": vps.LogLevel.INFO,
    "warning": vps.LogLevel.WARNING,
    "error": vps.LogLevel.ERROR,
}.get(cfg["logLevel"].lower(), vps.LogLevel.INFO))

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
            # Overhang the left domain edge: a box ending exactly on the
            # boundary leaves uncovered oxide cells there, which classify as
            # AMBIENT and admit unbarriered oxidant.  Same applies on the
            # right for a full-width mask (set maskEdge > xExtent for that).
            [-2.0 * x_extent, pad_oxide_thickness - mask_contact_eps],
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
if cfg["maskPermeabilityRatio"] > 0.0:
    model.setMaskPermeabilityRatio(cfg["maskPermeabilityRatio"])
model.setMaskBarrierOnly(cfg["maskBarrierOnly"])
if cfg["dealGroveB"] > 0.0 or cfg["dealGroveBoA"] > 0.0:
    model.setDealGroveRates(cfg["dealGroveB"], cfg["dealGroveBoA"])
model.setTimeStep(time_step)
model.setInitialTimeStep(cfg["initialTimeStep"])
if cfg["regridDelta"] > 0.0:
    model.setRegrid(cfg["regridDelta"])
model.setStressDependentOxidation(cfg["stressDependentOxidation"])
if cfg["reactionActivationVolume"] > 0.0:
    model.setReactionActivationVolume(cfg["reactionActivationVolume"])
if cfg["diffusionActivationVolume"] > 0.0:
    model.setDiffusionActivationVolume(cfg["diffusionActivationVolume"])
if cfg["shearActivationVolume"] >= 0.0:
    model.setShearActivationVolume(cfg["shearActivationVolume"])
model.setMaxGridPoints(max_grid_points)
model.setMechanicsIterations(cfg["mechanicsIterations"])
model.setCouplingIterations(cfg["couplingIterations"])
model.setCouplingTolerance(cfg["couplingTolerance"])
mask_params = vls.OxidationPresets.siliconNitrideMask(temperature)
mask_params.poissonRatio = cfg["maskPoissonRatio"]
mask_params.youngModulus = cfg["maskYoungModulus"]
if cfg["maskViscosity"] >= 0.0:
    mask_params.viscosity = cfg["maskViscosity"]
if cfg["maskShearActivationVolume"] >= 0.0:
    mask_params.shearActivationVolume = cfg["maskShearActivationVolume"]
model.setMaskParameters(mask_params)
model.setMechanicsTolerance(cfg["mechanicsTolerance"])

use_gpu = cfg["useGpu"].lower()
if use_gpu == "gpu":
    model.setGpuMode(vps.GpuMode.Gpu)
elif use_gpu == "auto":
    model.setGpuMode(vps.GpuMode.Auto)
elif use_gpu == "cpu":
    model.setGpuMode(vps.GpuMode.Cpu)

model.saveSurfaceMesh(domain, f"{output_prefix}_step_000.vtp")
model.saveVolumeMesh(domain, f"{output_prefix}_step_000")

model.setTime(oxidation_time)
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
