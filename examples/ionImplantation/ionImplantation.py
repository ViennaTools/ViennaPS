"""Masked implant + anneal with all parameters from config.txt.

The implant model uses an explicit dual-Pearson IV profile built from
moments provided in the config file (projectedRange, depthSigma, skewness,
kurtosis, lateralSigma, headFraction, tail* equivalents).

Anneal physics (diffusivity, I/V parameters, solid solubility, SCORE
partition factors) are loaded from the vsclib annealing CSV.  The
temperature schedule is read from the config file.

Usage:
    python implantMask2D.py [config.txt]
"""
import sys
from pathlib import Path

SPECIES_TO_DOPANT = {
    "B": "boron", "As": "arsenic", "P": "phosphorus", "Sb": "antimony",
    "In": "indium", "C": "carbon", "F": "fluorine", "N": "nitrogen", "Al": "aluminum",
}

DOSE_CONTROL_MAP = {
    "off": "Off",
    "waferdose": "WaferDose",
    "beamdose": "BeamDose",
}


def read_config(path: str) -> dict:
    params = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if "," in v:
                params[k] = [float(x.strip()) for x in v.split(",") if x.strip()]
            else:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return params


def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.txt"
    if not Path(cfg_path).is_file():
        print(f"Config not found: {cfg_path}")
        return 1

    params = read_config(cfg_path)
    print(f"--- ViennaPS Explicit Implant & Anneal (config: {cfg_path}) ---")

    try:
        import viennacs as _vcs
        import viennals as _vls
        _vcs.setDimension(2)
        _vls.setDimension(2)
        vcs = _vcs.d2
        vls = _vls.d2
        MaterialMap = _vls.MaterialMap
        BoundaryConditionEnum = _vls.BoundaryConditionEnum
        BooleanOperationEnum = _vls.BooleanOperationEnum
    except ImportError:
        try:
            import viennacs2d as vcs
            _vcs = vcs
            import viennals2d as vls
            MaterialMap = vls.MaterialMap
            BoundaryConditionEnum = vls.BoundaryConditionEnum
            BooleanOperationEnum = vls.BooleanOperationEnum
        except ImportError as exc:
            raise SystemExit(
                "Missing ViennaCS/ViennaLS Python modules. Install matching "
                "pairs (`viennacs` + `viennals`) or legacy (`viennacs2d` + "
                "`viennals2d`), then rerun."
            ) from exc

    # -------------------------------------------------------------------------
    # 1. Geometry
    # -------------------------------------------------------------------------
    grid_delta      = float(params.get("gridDelta", 0.25))
    x_extent        = float(params.get("xExtent", 100.0))
    top_space       = float(params.get("topSpace", 15.0))
    substrate_depth = float(params.get("substrateDepth", 100.0))
    opening_width   = float(params.get("openingWidth", 15.0))
    mask_height     = float(params.get("maskHeight", 10.0))
    oxide_thickness = float(params.get("screenOxideThickness",
                                       params.get("oxideThickness", 2.0)))
    screen_thickness = float(params.get("screenThickness", oxide_thickness))

    bounds = [
        -0.5 * x_extent, 0.5 * x_extent,
        -substrate_depth, top_space + oxide_thickness + mask_height,
    ]
    boundary_conditions = [
        BoundaryConditionEnum.REFLECTIVE_BOUNDARY,
        BoundaryConditionEnum.INFINITE_BOUNDARY,
    ]

    material_map = MaterialMap()
    level_sets = []

    def _domain():
        return vls.Domain(bounds, boundary_conditions, grid_delta)

    def _add(ls, mat_id):
        if level_sets:
            vls.BooleanOperation(ls, level_sets[-1], BooleanOperationEnum.UNION).apply()
        level_sets.append(ls)
        material_map.insertNextMaterial(mat_id)

    sub_bottom = _domain()
    vls.MakeGeometry(sub_bottom, vls.Plane([0.0, -substrate_depth], [0.0, 1.0])).apply()
    _add(sub_bottom, 1)
    sub_top = _domain()
    vls.MakeGeometry(sub_top, vls.Plane([0.0, 0.0], [0.0, 1.0])).apply()
    _add(sub_top, 1)

    oxide = _domain()
    vls.MakeGeometry(oxide, vls.Plane([0.0, oxide_thickness], [0.0, 1.0])).apply()
    _add(oxide, 3)

    mask = _domain()
    vls.MakeGeometry(mask, vls.Plane([0.0, oxide_thickness + mask_height], [0.0, 1.0])).apply()
    opening = _domain()
    vls.MakeGeometry(opening, vls.Box(
        [-0.5 * opening_width, oxide_thickness - grid_delta],
        [ 0.5 * opening_width, oxide_thickness + mask_height + grid_delta],
    )).apply()
    vls.BooleanOperation(mask, opening, BooleanOperationEnum.RELATIVE_COMPLEMENT).apply()
    _add(mask, 2)

    cell_set = vcs.DenseCellSet()
    cell_set.setCellSetPosition(True)
    cell_set.setCoverMaterial(0)
    cell_set.fromLevelSets(level_sets, material_map, top_space)
    cell_set.writeVTU("initial_ownvalues.vtu")

    # -------------------------------------------------------------------------
    # 2. Implant — explicit dual-Pearson IV from config moments
    # -------------------------------------------------------------------------
    species      = str(params.get("species", "B"))
    material     = str(params.get("material", "Si"))
    tilt_deg     = float(params.get("angle", 7.0))
    rotation_deg = float(params.get("rotationDeg", 0.0))
    dose_cm2     = float(params.get("doseCm2", 1e15))
    energy_kev   = float(params.get("energyKeV", 10.0))

    # Head Pearson IV moments (required for explicit mode)
    proj_range   = float(params["projectedRange"])
    depth_sigma  = float(params["depthSigma"])
    skewness     = float(params["skewness"])
    kurtosis     = float(params["kurtosis"])
    lat_mu       = float(params.get("lateralMu", 0.0))
    lat_sigma    = float(params.get("lateralSigma", 5.0))

    head_p = _vcs.PearsonIVParameters()
    head_p.mu    = proj_range
    head_p.sigma = depth_sigma
    head_p.beta  = skewness   # γ₁
    head_p.gamma = kurtosis   # γ₂

    if "headFraction" in params:
        # Dual-Pearson IV (head + channeling tail)
        head_fraction    = float(params["headFraction"])
        tail_proj_range  = float(params.get("tailProjectedRange", proj_range * 2.5))
        tail_depth_sigma = float(params.get("tailDepthSigma", depth_sigma * 2.5))
        tail_skewness    = float(params.get("tailSkewness", 0.0))
        tail_kurtosis    = float(params.get("tailKurtosis", 3.0))
        tail_lat_mu      = float(params.get("tailLateralMu", 0.0))
        tail_lat_sigma   = float(params.get("tailLateralSigma", lat_sigma))

        tail_p = _vcs.PearsonIVParameters()
        tail_p.mu    = tail_proj_range
        tail_p.sigma = tail_depth_sigma
        tail_p.beta  = tail_skewness
        tail_p.gamma = tail_kurtosis

        print(f"Implanting {species} into {material} at {energy_kev} keV, {tilt_deg}° tilt "
              f"(dual-Pearson IV, head fraction {head_fraction}) ...")
        model = vcs.ImplantDualPearsonIV(
            head_p, tail_p, head_fraction,
            lat_mu, lat_sigma,
            tail_lat_mu, tail_lat_sigma,
        )
    else:
        # Single Pearson IV
        print(f"Implanting {species} into {material} at {energy_kev} keV, {tilt_deg}° tilt "
              f"(Pearson IV) ...")
        model = vcs.ImplantPearsonIV(head_p, lat_mu, lat_sigma)

    # Damage: table lookup (damage moments not specified in config.txt)
    damage_recipe = _vcs.DamageRecipe()
    damage_recipe.species        = species
    damage_recipe.material       = material
    damage_recipe.energyKeV      = energy_kev
    damage_recipe.tiltDeg        = tilt_deg
    damage_recipe.rotationDeg    = rotation_deg
    damage_recipe.dosePerCm2     = dose_cm2
    damage_recipe.screenThickness = screen_thickness
    damage_model = vcs.RecipeDrivenDamageModel(damage_recipe)

    implant = vcs.Implant()
    implant.setCellSet(cell_set)
    implant.setImplantModel(model)
    implant.setDamageModel(damage_model)
    implant.setImplantAngle(tilt_deg)
    implant.setDose(dose_cm2)
    implant.setLengthUnitInCm(1.0e-7)  # nm -> cm for dose scaling parity with C++
    dose_control = str(params.get("doseControl", "WaferDose")).strip().lower()
    dose_control_name = DOSE_CONTROL_MAP.get(dose_control, "WaferDose")
    implant.setDoseControl(getattr(_vcs.ImplantDoseControl, dose_control_name))
    implant.setMaskMaterials([2])
    implant.setScreenMaterials([3])
    implant.apply()
    cell_set.writeVTU("post_implant_ownvalues.vtu")

    # -------------------------------------------------------------------------
    # 3. Anneal — physics from vsclib CSV; schedule from config
    # -------------------------------------------------------------------------
    durations = params.get("annealStepDurations", [])
    temps     = params.get("annealTemperatures", [])

    peak_T = (max(temps) if temps
              else float(params.get("annealTemperature", 1323.15)))

    print(f"Annealing: peak T = {peak_T - 273.15:.1f} °C ...")

    dopant_name = SPECIES_TO_DOPANT.get(species, species.lower())

    print("\n--- ANNEAL PARAMETERS (PYTHON) ---")
    print(f"Dopant Name:         {dopant_name}")
    print(f"Peak Temperature:    {peak_T} K")
    print(f"Durations:           {durations}")
    print(f"Temperatures:        {temps}")
    print(f"Defect Coupling:     {bool(int(params.get('annealDefectCoupling', 1)))}")
    print(f"Diffusion Materials: [1] (Si)")
    print(f"Blocking Materials:  [2, 3] (Mask, SiO2)")
    try:
        c_field = cell_set.getScalarData("concentration")
        print(f"Max Concentration:   {max(c_field):.4e} (entering anneal)")
        d_field = cell_set.getScalarData("Damage")
        if d_field: print(f"Max Damage:          {max(d_field):.4e} (entering anneal)")
    except Exception:
        pass
    print("----------------------------------\n")

    print("\n--- ANNEALING MODEL PARAMETERS (from vsclib CSV) ---")
    try:
        import csv, os, math
        vsclib_dir = os.environ.get("VSCLIB_DIR", "")
        if not vsclib_dir:
            vsclib_dir = os.path.join(os.path.dirname(_vcs.__file__), "vsclib")
        csv_path = os.path.join(vsclib_dir, "anneal", "annealing.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("material", "").strip().lower() == "silicon" and \
                       row.get("species", "").strip().lower() == dopant_name.lower():
                        if row.get("D0_cm2_per_s", "").strip():
                            print(f"Found parameters for {dopant_name} in silicon:")
                            for k, v in row.items():
                                if v.strip(): print(f"  {k}: {v}")
                            try:
                                kB = 8.617333262145e-5
                                D0 = float(row.get("D0_cm2_per_s", 0))
                                Ea = float(row.get("Ea_eV", 0))
                                D_T = D0 * math.exp(-Ea / (kB * peak_T))
                                print(f"  -> Evaluated Diffusivity at {peak_T} K: {D_T:.4e} cm^2/s")
                            except Exception: pass
                            break
    except Exception as e: print(f"Error reading CSV: {e}")
    print("----------------------------------------------------\n")

    anneal = vcs.Anneal()
    anneal.setCellSet(cell_set)
    anneal.setSpeciesLabel("concentration")
    anneal.setDopantName(dopant_name)
    anneal.setTemperature(peak_T)        # peak T for loadAnnealingCSV Arrhenius + activation
    anneal.loadAnnealingCSV()            # diffusivity, I/V params, solid solubility, SCORE factors

    mode = str(params.get("annealMode", "implicit")).lower()
    anneal_mode_enum = getattr(_vcs, "DiffusionSolverMode", None)
    if anneal_mode_enum is None:
        anneal_mode_enum = getattr(_vcs, "AnnealMode", None)
    if anneal_mode_enum is not None and hasattr(anneal, "setMode"):
        if mode == "implicit":
            anneal.setMode(anneal_mode_enum.GaussSeidel)
        elif mode == "explicit":
            anneal.setMode(anneal_mode_enum.Explicit)

    anneal.setTemperatureSchedule(durations, temps)

    if not durations:
        anneal.setDuration(float(params.get("annealDuration", 5.0)))

    anneal.setDiffusionMaterials([1])
    anneal.setBlockingMaterials([2, 3])

    if int(params.get("annealDefectCoupling", 1)):
        anneal.enableDefectCoupling(True)

    anneal.apply()
    cell_set.writeVTU("post_anneal_ownvalues.vtu")

    print("Done.")
    print("  initial_ownvalues.vtu      : geometry")
    print("  post_implant_ownvalues.vtu : concentration + damage fields")
    print("  post_anneal_ownvalues.vtu  : concentration (total) + active_concentration + I/V fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
