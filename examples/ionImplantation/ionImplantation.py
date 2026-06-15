"""
Masked ion implantation + anneal — Python example.

Mirrors ionImplantation.cpp:  builds a 2-D masked-substrate domain, stamps
a dual-Pearson IV dopant profile + Hobler damage profile into the cell set,
then diffuses / activates with a 3-step ramp anneal schedule.

All parameters are read from config.txt (manual/analytic Pearson IV mode).
The same parameters work with the config_default.txt file if the table-driven
mode is preferred (set useTableModel = True below).

Usage
-----
python ionImplantation.py              # uses config.txt (manual Pearson IV)
python ionImplantation.py config.txt   # same, explicit path
python ionImplantation.py config_default.txt  # table-driven (requires modeldb)

Output VTU files:
  initial.vtu       — geometry + material IDs
  post_implant.vtu  — P_total + P_damage
  post_anneal.vtu   — P_total + P_active + interstitial/vacancy fields

Requirements
------------
  import viennaps  (ViennaPS Python bindings, compiled with VIENNAPS_BUILD_PYTHON=ON)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import viennaps as ps
    import viennals as ls
except ImportError as exc:
    raise SystemExit(
        "Could not import viennaps.  Build ViennaPS with "
        "-DVIENNAPS_BUILD_PYTHON=ON and install the package first.\n"
        f"Original error: {exc}"
    )

ps.setDimension(2)
ls.setDimension(2)
vps = ps.d2
vls = ls


def read_config(path: str) -> dict[str, str]:
    """Parse a simple key=value config file (lines starting with # ignored)."""
    params: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def get(p: dict, key: str, default=None):
    """Get a float from the config dict with an optional default."""
    if key in p:
        return float(p[key])
    if default is not None:
        return float(default)
    raise KeyError(f"Required config key '{key}' not found.")


def parse_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def canonical_species_name(species: str) -> str:
    return {
        "b": "boron",
        "p": "phosphorus",
        "as": "arsenic",
        "sb": "antimony",
        "ge": "germanium",
        "c": "carbon",
        "n": "nitrogen",
        "f": "fluorine",
    }.get(species.lower(), species.lower())


def canonical_material_name(material: str) -> str:
    return {
        "si": "silicon",
        "bulk_si": "silicon",
        "bulksi": "silicon",
        "sio2": "oxide",
        "oxide": "oxide",
        "si3n4": "nitride",
        "sin": "nitride",
    }.get(material.lower(), material.lower())


def arrhenius(prefactor: float, ea_ev: float, temperature_k: float) -> float:
    kb_ev = 8.617333262145e-5
    return max(prefactor, 0.0) * math.exp(-max(ea_ev, 0.0) / (kb_ev * max(temperature_k, 1.0)))


def thermal_average_arrhenius(prefactor: float, ea_ev: float,
                              durations: list[float],
                              temperatures: list[float],
                              fallback_temperature_k: float) -> float:
    if not durations or not temperatures:
        return arrhenius(prefactor, ea_ev, fallback_temperature_k)

    total = 0.0
    weighted = 0.0
    if len(temperatures) == len(durations):
        for dt, temp in zip(durations, temperatures):
            dt = max(dt, 0.0)
            total += dt
            weighted += dt * arrhenius(prefactor, ea_ev, temp)
    elif len(temperatures) == len(durations) + 1:
        samples = 16
        for i, dt in enumerate(durations):
            dt = max(dt, 0.0)
            total += dt
            segment = 0.0
            for j in range(samples):
                a = (j + 0.5) / samples
                temp = temperatures[i] * (1.0 - a) + temperatures[i + 1] * a
                segment += arrhenius(prefactor, ea_ev, temp)
            weighted += dt * segment / samples
    else:
        return arrhenius(prefactor, ea_ev, fallback_temperature_k)

    return weighted / total if total > 0.0 else arrhenius(prefactor, ea_ev, fallback_temperature_k)


def load_anneal_modeldb(cfg: dict[str, str], modeldb_root: Path,
                        durations: list[float],
                        temperatures: list[float],
                        peak_temperature_k: float) -> dict[str, float]:
    """Read the small subset of modeldb/anneal used by this example."""
    species_name = canonical_species_name(cfg.get("species", "P"))
    material_name = canonical_material_name(cfg.get("material", "Si"))
    table = modeldb_root / "anneal" / "annealing.csv"

    values: dict[str, float] = {
        "annealRecombinationRate": 0.0,
        "annealInterstitialSinkRate": 0.0,
        "annealVacancySinkRate": 0.0,
        "annealScoreIFactor": 0.5,
        "annealScoreVFactor": 0.5,
    }
    d_total = d_int = d_vac = None

    def thermal(prefactor: float, ea_ev: float) -> float:
        return thermal_average_arrhenius(
            prefactor, ea_ev, durations, temperatures, peak_temperature_k
        )

    with table.open() as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 6 or cols[0].lower() == "material":
                continue
            material, category, name, param = (c.lower() for c in cols[:4])
            if material != material_name:
                continue
            try:
                prefactor = float(cols[4])
                ea_ev = float(cols[5])
            except ValueError:
                continue

            if category == "dopant" and name == species_name:
                if param == "d":
                    d_total = (prefactor, ea_ev)
                elif param == "int_d":
                    d_int = (prefactor, ea_ev)
                elif param == "vac_d":
                    d_vac = (prefactor, ea_ev)
                elif param == "score_ifactor":
                    values["annealScoreIFactor"] = prefactor
                elif param == "score_vfactor":
                    values["annealScoreVFactor"] = prefactor
                elif param == "solubility":
                    values["annealSolidSolubilityC0"] = prefactor * 1e-21
                    values["annealSolidSolubilityEa"] = ea_ev
            elif category == "defect" and name == "interstitial" and param == "di":
                values["annealInterstitialDiffusivity"] = thermal(prefactor, ea_ev)
            elif category == "defect" and name == "vacancy" and param == "dv":
                values["annealVacancyDiffusivity"] = thermal(prefactor, ea_ev)
            elif category == "defect" and name == "interstitial" and param == "cstar":
                values["annealInterstitialEqC0"] = prefactor * 1e-21
                values["annealInterstitialEqEa"] = ea_ev
            elif category == "defect" and name == "vacancy" and param == "cstar":
                values["annealVacancyEqC0"] = prefactor * 1e-21
                values["annealVacancyEqEa"] = ea_ev
            elif category == "defect" and name == "icluster" and param == "ikfc":
                values["annealClusterKfc"] = thermal(prefactor * 1e-21, ea_ev)
            elif category == "defect" and name == "icluster" and param == "ikr":
                values["annealClusterKr"] = thermal(prefactor, ea_ev)
            elif category == "defect" and name == "icluster" and param == "initpercent":
                values["annealClusterInitFraction"] = prefactor / 100.0
            elif category == "property" and param == "krec":
                values["annealRecombinationRate"] = thermal(prefactor * 1e-21, ea_ev)

    selected = d_total
    if selected is None and (d_int is not None or d_vac is not None):
        selected = d_int if d_vac is None or (d_int is not None and d_int[0] >= d_vac[0]) else d_vac
    if selected is None:
        raise RuntimeError(f"No anneal diffusivity entry for {species_name} in {material_name}: {table}")

    values["annealD0"] = selected[0]
    values["annealEa"] = selected[1]
    return values


def print_implant_material_summary(domain, label: str) -> None:
    cs = domain.getCellSet()
    concentration = cs.getScalarData(label)
    materials = cs.getScalarData("Material")
    if concentration is None or materials is None:
        return

    material_names = {
        int(ps.Material.Air): "Air",
        int(ps.Material.Mask): "Mask",
        int(ps.Material.Si): "Si",
        int(ps.Material.SiO2): "SiO2",
    }
    by_material: dict[int, tuple[float, float]] = {}
    for mat, conc in zip(materials, concentration):
        if conc <= 0.0:
            continue
        key = int(mat)
        max_val, sum_val = by_material.get(key, (0.0, 0.0))
        by_material[key] = (max(max_val, conc), sum_val + conc)

    print("\n--- IMPLANT MATERIAL SUMMARY (Python) ---")
    if not by_material:
        print(f"  {label}: no positive concentration")
    for key, (max_val, sum_val) in sorted(by_material.items()):
        print(
            f"  {material_names.get(key, f'Material {key}')}: "
            f"{label} max={max_val:.6g}  sum={sum_val:.6g}"
        )
    print("-----------------------------------------")


def run(cfg_path: str) -> None:
    cfg = read_config(cfg_path)
    if not cfg:
        sys.exit(f"Config not found: {cfg_path}")

    # The table-backed implant path is currently only reliable when the Python
    # example runs single-threaded. Keep that as the default for smoke tests and
    # allow configs to opt into more threads once the backend is thread-clean.
    if hasattr(ps, "setNumThreads"):
        ps.setNumThreads(max(1, int(get(cfg, "numThreads", 1))))

    use_table = "projectedRange" not in cfg
    modeldb_root = Path(__file__).resolve().parents[2] / "modeldb"
    print(f"--- ViennaPS {'Table-Driven' if use_table else 'Explicit'} "
          f"Implant & Anneal (config: {cfg_path}) ---")

    # ── Geometry ──────────────────────────────────────────────────────────────
    grid_delta       = get(cfg, "gridDelta")
    x_extent         = get(cfg, "xExtent")
    top_space        = get(cfg, "topSpace")
    substrate_depth  = get(cfg, "substrateDepth")
    opening_width    = get(cfg, "openingWidth")
    mask_height      = get(cfg, "maskHeight")
    oxide_thickness  = get(cfg, "screenOxideThickness",
                           cfg.get("oxideThickness", "2.0"))
    screen_thickness = get(cfg, "screenThickness", oxide_thickness)

    bounds = [-0.5 * x_extent, 0.5 * x_extent,
              -substrate_depth,
              top_space + oxide_thickness + mask_height]

    bc = [ls.BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
          ls.BoundaryConditionEnum.INFINITE_BOUNDARY]

    domain = vps.Domain(bounds, bc, grid_delta)

    def makels():
        return vls.Domain(bounds, bc, grid_delta)

    # Si substrate bottom
    level_set = makels()
    vls.MakeGeometry(level_set, vls.Plane([0., -substrate_depth], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(level_set, ps.Material.Si)

    # Si substrate top (surface at y = 0)
    level_set = makels()
    vls.MakeGeometry(level_set, vls.Plane([0., 0.], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(level_set, ps.Material.Si)

    # Screen oxide (y = 0 to y = oxide_thickness)
    level_set = makels()
    vls.MakeGeometry(level_set, vls.Plane([0., oxide_thickness], [0., 1.])).apply()
    domain.insertNextLevelSetAsMaterial(level_set, ps.Material.SiO2)

    # Hard mask with opening
    level_set = makels()
    vls.MakeGeometry(
        level_set, vls.Plane([0., oxide_thickness + mask_height], [0., 1.])
    ).apply()
    domain.insertNextLevelSetAsMaterial(level_set, ps.Material.Mask)

    window = makels()
    vls.MakeGeometry(
        window,
        vls.Box(
            [-0.5 * opening_width, oxide_thickness - grid_delta],
            [0.5 * opening_width, oxide_thickness + mask_height + grid_delta],
        ),
    ).apply()
    domain.applyBooleanOperation(
        window, ls.BooleanOperationEnum.RELATIVE_COMPLEMENT)

    # Cell set
    domain.generateCellSet(top_space, ps.Material.Air, True)
    domain.getCellSet().buildNeighborhood()

    out_suffix = "_preset.vtu" if use_table else "_manual.vtu"
    domain.getCellSet().writeVTU("initial" + out_suffix)

    dose       = get(cfg, "doseCm2")
    tilt_angle = get(cfg, "angle", 7.)
    species    = cfg.get("species", "P")

    # ── Implant model ─────────────────────────────────────────────────────────
    if use_table:
        species_name = canonical_species_name(species)
        material_name = canonical_material_name(cfg.get("material", "Si"))
        substrate = cfg.get("substrateType", "crystalline").lower()
        implant_table = (
            modeldb_root / "implant" /
            f"{species_name}_in_{material_name}_{substrate}.csv"
        )
        damage_table = (
            modeldb_root / "damage" /
            f"{species_name}_damage_in_{material_name}.csv"
        )
        implant_model = vps.ImplantTableModel(
            str(implant_table), species_name, material_name, substrate,
            get(cfg, "energyKeV"), tilt_angle, get(cfg, "rotationDeg", 0.),
            dose, screen_thickness, get(cfg, "damageLevel", 0.),
        )
        damage_model = vps.DamageTableModel(
            str(damage_table), species_name, material_name,
            get(cfg, "energyKeV"), tilt_angle, get(cfg, "rotationDeg", 0.),
            dose, screen_thickness,
        )
        description = (
            f"{species} into {cfg.get('material','Si')} at "
            f"{get(cfg,'energyKeV'):.0f} keV, {tilt_angle:.0f} deg tilt "
            f"(table model)"
        )
    else:
        # Build PearsonIVParameters from config (manual mode). These moments are
        # substrate-depth moments; do not subtract screen oxide thickness.
        head_params = ps.PearsonIVParameters()
        head_params.mu    = get(cfg, "projectedRange")
        head_params.sigma = get(cfg, "depthSigma")
        head_params.beta  = get(cfg, "skewness")    # → C++ params.beta (β₂ position)
        head_params.gamma = get(cfg, "kurtosis")    # → C++ params.gamma (γ₁ position)
        lateral_mu        = get(cfg, "lateralMu", 0.)
        lateral_sigma     = get(cfg, "lateralSigma", 5.)

        head_fraction = get(cfg, "headFraction", -1.)
        if head_fraction > 0.:
            # Dual Pearson IV (crystalline Si with channeling tail)
            tail_params = ps.PearsonIVParameters()
            tail_params.mu    = get(cfg, "tailProjectedRange")
            tail_params.sigma = get(cfg, "tailDepthSigma")
            tail_params.beta  = get(cfg, "tailSkewness")
            tail_params.gamma = get(cfg, "tailKurtosis")
            tail_lateral_mu    = get(cfg, "tailLateralMu", 0.)
            tail_lateral_sigma = get(cfg, "tailLateralSigma", lateral_sigma)

            implant_model = vps.ImplantDualPearsonIV(
                head_params, tail_params, head_fraction,
                lateral_mu, lateral_sigma,
                tail_lateral_mu, tail_lateral_sigma,
            )
            description = (
                f"{species} into {cfg.get('material','Si')} at "
                f"{get(cfg,'energyKeV'):.0f} keV, "
                f"{tilt_angle:.0f} deg tilt "
                f"(dual-Pearson IV, head fraction {head_fraction:.4f})"
            )
        else:
            # Single Pearson IV
            implant_model = vps.ImplantPearsonIV(head_params, lateral_mu, lateral_sigma)
            description = (
                f"{species} into {cfg.get('material','Si')} at "
                f"{get(cfg,'energyKeV'):.0f} keV, "
                f"{tilt_angle:.0f} deg tilt (single Pearson IV)"
            )

        # Damage model (Hobler)
        damage_model = vps.ImplantDamageHobler(
            get(cfg, "damageProjectedRange"),
            get(cfg, "damageVerticalSigma"),
            get(cfg, "damageLambda"),
            get(cfg, "damageDefectsPerIon"),
            get(cfg, "damageLateralSigma"),
            get(cfg, "damageLateralDeltaSigma", 0.),
        )

    label_total        = f"{species}_total"
    label_active       = f"{species}_active"
    label_damage       = f"{species}_damage"
    label_interstitial = f"{species}_interstitial"
    label_vacancy      = f"{species}_vacancy"

    implant = vps.IonImplantation()
    implant.setImplantModel(implant_model)
    implant.setDamageModel(damage_model)
    implant.setDose(dose)
    implant.setTiltAngle(tilt_angle)
    implant.setLengthUnit(1e-7)   # nm → cm
    implant.setDoseControl(ps.ImplantDoseControl.WaferDose)
    implant.setMaskMaterials([ps.Material.Mask])
    implant.setScreenMaterials([ps.Material.SiO2])
    implant.setConcentrationLabel(label_total)
    implant.setDamageLabel(label_damage)
    implant.setLastDamageLabel(label_damage + "_last")

    print(f"Implanting {description} ...")

    # ── Anneal model ──────────────────────────────────────────────────────────
    durations    = parse_list(cfg.get("annealStepDurations", "9,5,9"))
    temperatures = parse_list(cfg.get("annealTemperatures", "873.15,1323.15,1323.15,873.15"))
    peak_T       = max(temperatures)
    anneal_source = cfg.get(
        "annealParameterSource", "modeldb" if use_table else "manual"
    ).lower()
    required_anneal_keys = {
        "annealD0",
        "annealEa",
        "annealInterstitialDiffusivity",
        "annealVacancyDiffusivity",
        "annealSolidSolubilityC0",
        "annealSolidSolubilityEa",
    }
    use_anneal_modeldb = (
        anneal_source not in {"manual", "config", "user"}
        or not required_anneal_keys.issubset(cfg)
    )
    anneal_defaults = (
        load_anneal_modeldb(cfg, modeldb_root, durations, temperatures, peak_T)
        if use_anneal_modeldb else {}
    )

    def aget(key: str, default=None) -> float:
        if key in cfg:
            return get(cfg, key)
        if key in anneal_defaults:
            return anneal_defaults[key]
        if default is not None:
            return float(default)
        raise KeyError(f"Required anneal key '{key}' not found.")

    anneal = vps.Anneal()
    anneal.setTemperatureSchedule(durations, temperatures)
    anneal.setArrheniusParameters(aget("annealD0"), aget("annealEa"))
    anneal.setMode(ps.AnnealMode.GaussSeidel)
    anneal.setImplicitSolverOptions(
        int(aget("annealImplicitMaxIterations", 400)),
        aget("annealImplicitTolerance", 1e-6),
        aget("annealImplicitRelaxation", 1.0),
    )
    anneal.setDiffusionMaterials([ps.Material.Si])
    anneal.setBlockingMaterials([ps.Material.Mask, ps.Material.SiO2])
    anneal.setSpeciesLabel(label_total)
    anneal.setActiveLabel(label_active)

    # Defect coupling
    if int(get(cfg, "annealDefectCoupling", 1)):
        anneal.enableDefectCoupling(True)
        anneal.setDamageLabels(label_damage, label_damage + "_last")
        anneal.setDefectLabels(label_interstitial, label_vacancy)
        anneal.setDefectDiffusivities(
            aget("annealInterstitialDiffusivity"),
            aget("annealVacancyDiffusivity"),
        )
        anneal.setDefectReactionRates(
            aget("annealRecombinationRate", 0.),
            aget("annealInterstitialSinkRate", 0.),
            aget("annealVacancySinkRate", 0.),
        )
        score_i = aget("annealScoreIFactor", 0.5)
        score_v = aget("annealScoreVFactor", 0.5)
        if score_i + score_v > 0:
            anneal.setDefectPartition(
                score_i / (score_i + score_v),
                score_v / (score_i + score_v),
            )

    # Solid activation
    if int(get(cfg, "annealSolidActivation", 1)):
        anneal.enableSolidActivation(True)
        anneal.setSolidSolubilityArrhenius(
            aget("annealSolidSolubilityC0"),
            aget("annealSolidSolubilityEa"),
        )

    # Defect equilibrium (optional)
    if "annealInterstitialEqC0" in cfg or "annealInterstitialEqC0" in anneal_defaults:
        anneal.enableDefectEquilibrium(True)
        anneal.setDefectEquilibriumArrhenius(
            aget("annealInterstitialEqC0"),
            aget("annealInterstitialEqEa"),
            aget("annealVacancyEqC0"),
            aget("annealVacancyEqEa"),
        )

    # Defect clustering (optional)
    if any(k in cfg or k in anneal_defaults for k in ("annealClusterKfc", "annealClusterKr")):
        anneal.enableDefectClustering(True)
        anneal.setDefectClusterKinetics(
            aget("annealClusterKfi", 0.),
            aget("annealClusterKfc", 0.),
            aget("annealClusterKr", 0.),
        )
        anneal.setDefectClusterInitFraction(
            aget("annealClusterInitFraction", 0.01)
        )

    print(f"Annealing: peak T = {peak_T - 273.15:.0f} C ...")

    # ── Run ───────────────────────────────────────────────────────────────────
    vps.Process(domain, implant, 0.).apply()
    if int(get(cfg, "debugMaterialSummary", 0)):
        print_implant_material_summary(domain, label_total)
    domain.getCellSet().writeVTU("post_implant" + out_suffix)

    vps.Process(domain, anneal, 0.).apply()
    # Refresh the active dopant field after the full thermal step so downstream
    # sheet-resistance/net-doping tools see the post-anneal concentration.
    anneal.applyActivation(domain)
    if int(get(cfg, "debugMaterialSummary", 0)):
        print_implant_material_summary(domain, label_total)
    domain.getCellSet().writeVTU("post_anneal" + out_suffix)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n--- POST-ANNEAL STATS (Python) ---")
    cs = domain.getCellSet()
    for label in [label_total, label_active, label_damage,
                  label_interstitial, label_vacancy]:
        field = cs.getScalarData(label)
        if field is None:
            print(f"  {label}: <missing>")
            continue
        max_val = max(field)
        sum_val = sum(field)
        print(f"  {label}: max={max_val:.6g}  sum={sum_val:.6g}")
    print("-----------------------------------")

    print("\nDone.")
    print(f"  initial{out_suffix}      : geometry + material IDs")
    print(f"  post_implant{out_suffix} : {label_total} + {label_damage}")
    print(f"  post_anneal{out_suffix}  : {label_total} + {label_active} + I/V fields")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
    run(cfg_path)
