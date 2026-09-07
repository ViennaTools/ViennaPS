"""
P implantation in Si — analytic (manual) Pearson IV mode, Python version.

Direct Python port of pImplantManual.cpp.  All Pearson IV moments, damage
parameters, and anneal physics are read from config.txt — the same file whose
parameter values were populated from the ViennaPS modeldb tables.

Usage
-----
python pImplantManual.py              # reads config.txt in the current directory
python pImplantManual.py config.txt   # explicit path

Output VTU files (open in ParaView):
  initial.vtu        — geometry + material IDs
  post_implant.vtu   — P_total + P_damage
  post_anneal.vtu    — P_total + P_active + interstitial/vacancy fields

Output CSV depth profiles:
  profile_post_implant.csv — dopant + damage vs depth  (sum and peak across x)
  profile_post_anneal.csv  — dopant total/active + I/V vs depth

Requirements
------------
  import viennaps  (build ViennaPS with -DVIENNAPS_BUILD_PYTHON=ON)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

try:
    import viennaps.d2 as vps
    import viennaps as _vps_core
    import viennals.d2 as vls
    import viennals as _vls_core
except ImportError as exc:
    raise SystemExit(
        "Could not import viennaps.  Build ViennaPS with "
        "-DVIENNAPS_BUILD_PYTHON=ON and install the package.\n"
        f"Original error: {exc}"
    )


# ─── Config helpers ───────────────────────────────────────────────────────────

def read_config(path: str) -> dict[str, str]:
    """Parse a simple key=value config file (# = comment)."""
    params: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def get(p: dict, key: str, default=None) -> float:
    if key in p:
        return float(p[key])
    if default is not None:
        return float(default)
    raise KeyError(f"Required config key '{key}' not found.")


def parse_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


# ─── Depth profile ────────────────────────────────────────────────────────────

def write_depth_profile(domain, labels: list[str], filename: str,
                        depth_axis: int = 1,
                        surface_position: float = 0.0) -> None:
    """
    Write a CSV with the vertical depth profile of the given cell-set fields.

    For each depth slice (binned by positive depth into the substrate), two
    columns are written per field:
      ``<label>_sum`` — lateral sum across all x-cells (integrates the dose)
      ``<label>_max`` — peak value across x (matches a 1-D full-dose profile)

    Parameters
    ----------
    domain      : viennaps Domain object
    labels      : list of cell-set scalar field names to include
    filename    : output CSV path
    depth_axis  : cell-centre axis index for depth (1 = y for 2-D, default)
    """
    cs    = domain.getCellSet()
    n     = cs.getNumberOfCells()
    delta = cs.getGridDelta()

    # Retrieve scalar data vectors (None if a field hasn't been written yet)
    fields: list[list[float] | None] = []
    for lbl in labels:
        data = cs.getScalarData(lbl)
        fields.append(data if data is not None else None)

    # Accumulate per depth-bin:
    # {depth_rounded -> {"count": int, "sum": [...], "max": [...]}}
    bins: dict[float, dict] = {}
    n_fields = len(labels)

    for idx in range(n):
        center = cs.getCellCenter(idx)
        # ViennaPS geometry uses y < 0 inside Si. Report SIMS-style positive
        # depth into the substrate and skip cells above the wafer surface.
        depth = surface_position - center[depth_axis]
        if depth < 0:
            continue
        depth = round(depth / delta) * delta

        if depth not in bins:
            bins[depth] = {"count": 0,
                           "sum": [0.0] * n_fields,
                           "max": [0.0] * n_fields}
        b = bins[depth]
        b["count"] += 1
        for f, data in enumerate(fields):
            if data is None:
                continue
            v = float(data[idx])
            b["sum"][f] += v
            if v > b["max"][f]:
                b["max"][f] = v

    with open(filename, "w") as out:
        out.write("# Vertical depth profile\n")
        out.write("# depth_nm: positive distance into substrate"
                  " (surfacePosition - cell-centre coordinate)\n")
        out.write("# _sum: lateral sum at this depth\n")
        out.write("# _max: peak value across lateral dimension\n")
        header = "depth_nm"
        for lbl in labels:
            header += f",{lbl}_sum,{lbl}_max"
        out.write(header + "\n")

        for depth in sorted(bins):
            b = bins[depth]
            row = str(depth)
            for f in range(n_fields):
                row += f",{b['sum'][f]},{b['max'][f]}"
            out.write(row + "\n")

    print(f"  wrote: {filename} ({len(bins)} depth slices)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(cfg_path: str) -> None:
    cfg = read_config(cfg_path)
    if not cfg:
        sys.exit(f"Config not found: {cfg_path}")

    print(f"--- ViennaPS Manual Implant & Anneal (config: {cfg_path}) ---")

    # ── Geometry ──────────────────────────────────────────────────────────────
    grid_delta      = get(cfg, "gridDelta")
    x_extent        = get(cfg, "xExtent")
    top_space       = get(cfg, "topSpace")
    substrate_depth = get(cfg, "substrateDepth")
    opening_width   = get(cfg, "openingWidth")
    mask_height     = get(cfg, "maskHeight")
    oxide_thickness = get(cfg, "oxideThickness")
    screen_thickness = get(cfg, "screenThickness", oxide_thickness)

    bounds = [-0.5 * x_extent, 0.5 * x_extent,
              -substrate_depth,
              top_space + oxide_thickness + mask_height]
    bc = [_vls_core.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
          _vls_core.BoundaryConditionEnum.INFINITE_BOUNDARY]

    domain = vps.Domain(bounds, bc, grid_delta)

    def makels():
        return vls.Domain(bounds, bc, grid_delta)

    # Si substrate bottom
    ls = makels()
    vls.MakeGeometry(ls, vls.Plane([0., -substrate_depth], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(ls, _vps_core.Material.Si)

    # Si substrate top (surface at y = 0)
    ls = makels()
    vls.MakeGeometry(ls, vls.Plane([0., 0.], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(ls, _vps_core.Material.Si)

    # Screen oxide (y = 0 → y = oxide_thickness)
    ls = makels()
    vls.MakeGeometry(ls, vls.Plane([0., oxide_thickness], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(ls, _vps_core.Material.SiO2)

    # Hard mask with opening (cut via Boolean RELATIVE_COMPLEMENT)
    ls = makels()
    vls.MakeGeometry(
        ls, vls.Plane([0., oxide_thickness + mask_height], [0., 1.])
    ).apply()
    domain.insertNextLevelSetAsMaterial(ls, _vps_core.Material.Mask)

    window = makels()
    vls.MakeGeometry(
        window,
        vls.Box(
            [-0.5 * opening_width, oxide_thickness - grid_delta],
            [0.5 * opening_width, oxide_thickness + mask_height + grid_delta],
        ),
    ).apply()
    domain.applyBooleanOperation(
        window, _vls_core.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    domain.generateCellSet(top_space, _vps_core.Material.Air, True)
    domain.getCellSet().buildNeighborhood()
    domain.getCellSet().writeVTU("initial.vtu")

    # ── Implant model (dual Pearson IV from config) ───────────────────────────
    species = cfg.get("species", "P")
    label_total        = f"{species}_total"
    label_active       = f"{species}_active"
    label_damage       = f"{species}_damage"
    label_interstitial = f"{species}_interstitial"
    label_vacancy      = f"{species}_vacancy"

    head = _vps_core.PearsonIVParameters()
    # Manual Pearson moments are substrate-depth moments; screen thickness is
    # represented by the geometry/screen material and is not subtracted here.
    head.mu    = get(cfg, "projectedRange")
    head.sigma = get(cfg, "depthSigma")
    head.beta  = get(cfg, "skewness")   # → C++ params.beta (β₂ position)
    head.gamma = get(cfg, "kurtosis")   # → C++ params.gamma (γ₁ position)

    lateral_mu    = get(cfg, "lateralMu", 0.)
    lateral_sigma = get(cfg, "lateralSigma", 5.)

    head_fraction = get(cfg, "headFraction", -1.)
    if head_fraction > 0.:
        tail = _vps_core.PearsonIVParameters()
        tail.mu    = get(cfg, "tailProjectedRange")
        tail.sigma = get(cfg, "tailDepthSigma")
        tail.beta  = get(cfg, "tailSkewness")
        tail.gamma = get(cfg, "tailKurtosis")
        tail_mu    = get(cfg, "tailLateralMu", 0.)
        tail_sigma = get(cfg, "tailLateralSigma", lateral_sigma)

        implant_model = vps.ImplantDualPearsonIV(
            head, tail, head_fraction,
            lateral_mu, lateral_sigma,
            tail_mu, tail_sigma,
        )
        description = (
            f"{species} into {cfg.get('material','Si')} at "
            f"{get(cfg,'energyKeV'):.0f} keV, "
            f"{get(cfg,'angle',0.):.0f} deg tilt "
            f"(dual-Pearson IV, head fraction {head_fraction:.4f})"
        )
    else:
        implant_model = vps.ImplantPearsonIV(head, lateral_mu, lateral_sigma)
        description = (
            f"{species} into {cfg.get('material','Si')} at "
            f"{get(cfg,'energyKeV'):.0f} keV "
            f"(single Pearson IV)"
        )

    damage_model = vps.ImplantDamageHobler(
        get(cfg, "damageProjectedRange"),
        get(cfg, "damageVerticalSigma"),
        get(cfg, "damageLambda"),
        get(cfg, "damageDefectsPerIon"),
        get(cfg, "damageLateralSigma"),
        get(cfg, "damageLateralDeltaSigma", 0.),
    )

    implant = vps.IonImplantation()
    implant.setImplantModel(implant_model)
    implant.setDamageModel(damage_model)
    implant.setDose(get(cfg, "doseCm2"))
    implant.setTiltAngle(get(cfg, "angle", 0.))
    implant.setLengthUnit(1e-7)                               # nm → cm
    implant.setDoseControl(_vps_core.ImplantDoseControl.WaferDose)
    implant.setMaskMaterials([_vps_core.Material.Mask])
    implant.setScreenMaterials([_vps_core.Material.SiO2])
    implant.setConcentrationLabel(label_total)
    implant.setDamageLabel(label_damage)
    implant.setLastDamageLabel(label_damage + "_last")

    print(f"Implanting {description} ...")

    # ── Anneal model ──────────────────────────────────────────────────────────
    durations    = parse_list(cfg.get("annealStepDurations", "9,5,9"))
    temperatures = parse_list(cfg.get("annealTemperatures",
                                      "873.15,1323.15,1323.15,873.15"))
    peak_T = max(temperatures)

    anneal = vps.Anneal()
    anneal.setTemperatureSchedule(durations, temperatures)
    anneal.setArrheniusParameters(get(cfg, "annealD0"), get(cfg, "annealEa"))
    anneal.setMode(_vps_core.AnnealMode.GaussSeidel)
    anneal.setImplicitSolverOptions(
        int(get(cfg, "annealImplicitMaxIterations", 400)),
        get(cfg, "annealImplicitTolerance", 1e-6),
        get(cfg, "annealImplicitRelaxation", 1.0),
    )
    anneal.setDiffusionMaterials([_vps_core.Material.Si])
    anneal.setBlockingMaterials([_vps_core.Material.Mask,
                                  _vps_core.Material.SiO2])
    anneal.setSpeciesLabel(label_total)
    anneal.setActiveLabel(label_active)

    # Defect coupling
    if int(get(cfg, "annealDefectCoupling", 1)):
        anneal.enableDefectCoupling(True)
        anneal.setDamageLabels(label_damage, label_damage + "_last")
        anneal.setDefectLabels(label_interstitial, label_vacancy)
        anneal.setDefectDiffusivities(
            get(cfg, "annealInterstitialDiffusivity"),
            get(cfg, "annealVacancyDiffusivity"),
        )
        anneal.setDefectReactionRates(
            get(cfg, "annealRecombinationRate", 0.),
            get(cfg, "annealInterstitialSinkRate", 0.),
            get(cfg, "annealVacancySinkRate", 0.),
        )
        si = get(cfg, "annealScoreIFactor", 0.5)
        sv = get(cfg, "annealScoreVFactor", 0.5)
        if si + sv > 0:
            anneal.setDefectPartition(si / (si + sv), sv / (si + sv))

    # Solid activation
    if int(get(cfg, "annealSolidActivation", 1)):
        anneal.enableSolidActivation(True)
        anneal.setSolidSolubilityArrhenius(
            get(cfg, "annealSolidSolubilityC0"),
            get(cfg, "annealSolidSolubilityEa"),
        )

    # Defect equilibrium (optional — enabled when Eq parameters are present)
    if "annealInterstitialEqC0" in cfg:
        anneal.enableDefectEquilibrium(True)
        anneal.setDefectEquilibriumArrhenius(
            get(cfg, "annealInterstitialEqC0"),
            get(cfg, "annealInterstitialEqEa"),
            get(cfg, "annealVacancyEqC0"),
            get(cfg, "annealVacancyEqEa"),
        )

    # Interstitial clustering (enabled when Kfc or Kr are present)
    if "annealClusterKfc" in cfg or "annealClusterKr" in cfg:
        anneal.enableDefectClustering(True)
        anneal.setDefectClusterKinetics(
            get(cfg, "annealClusterKfi", 0.),
            get(cfg, "annealClusterKfc", 0.),
            get(cfg, "annealClusterKr", 0.),
        )
        anneal.setDefectClusterInitFraction(
            get(cfg, "annealClusterInitFraction", 0.01)
        )

    print(f"Annealing: peak T = {peak_T - 273.15:.0f} C ...")

    # ── Run processes ─────────────────────────────────────────────────────────
    vps.Process(domain, implant, 0.).apply()
    domain.getCellSet().writeVTU("post_implant.vtu")
    write_depth_profile(domain,
                        [label_total, label_damage],
                        "profile_post_implant.csv")

    vps.Process(domain, anneal, 0.).apply()
    domain.getCellSet().writeVTU("post_anneal.vtu")
    write_depth_profile(domain,
                        [label_total, label_active, label_damage,
                         label_interstitial, label_vacancy],
                        "profile_post_anneal.csv")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n--- POST-ANNEAL STATS ---")
    cs = domain.getCellSet()
    for label in [label_total, label_active, label_damage,
                  label_interstitial, label_vacancy]:
        data = cs.getScalarData(label)
        if data is None:
            print(f"  {label}: <missing>")
            continue
        max_val = max(data)
        sum_val = sum(data)
        print(f"  {label}: max={max_val:.6g}  sum={sum_val:.6g}")
    print("-------------------------")

    print("\nDone.")
    print(f"  initial.vtu              : geometry + material IDs")
    print(f"  post_implant.vtu         : {label_total} + {label_damage}")
    print(f"  post_anneal.vtu          : {label_total} + {label_active} + I/V fields")
    print(f"  profile_post_implant.csv : depth profile after implant")
    print(f"  profile_post_anneal.csv  : depth profile after anneal")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
    run(cfg_path)
