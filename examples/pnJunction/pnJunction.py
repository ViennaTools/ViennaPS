"""
pnJunction.py — Lateral PN junction by sequential masked implantation.

All simulation parameters are read from config.txt (no hardcoded values).
The config file records the source of every parameter (modeldb table row,
SIMS calibration result, or literature value), so swapping in calibrated
values requires only editing config.txt, not this script.

Two implant steps, each with the opposite half of the domain masked:

   |<—— P opening (x < 0) ——>|<—— B opening (x > 0) ——>|
         P, 30 keV, 1e14 cm⁻²     B, 20 keV, 1e14 cm⁻²
   ———————————————————————————————————————————————————————
                     Si substrate

After a 1000 °C / 30 s anneal the metallurgical PN junction lies near x = 0.
NetDoping extracts the lateral junction x-position at the scan depth.
SheetResistance gives Rsh for both sides.

Usage
-----
  python pnJunction.py                     # reads config.txt in current dir
  python pnJunction.py path/to/config.txt  # explicit config path
  python pnJunction.py --no-vtk            # skip VTK output
  python pnJunction.py --plot              # show concentration plots (requires matplotlib)

Requires ViennaPS Python bindings built with -DVIENNAPS_BUILD_PYTHON=ON.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import viennaps as ps

ps.setDimension(2)


# ── Build a flat Si substrate (no physical mask) ───────────────────────────────
def build_substrate(cfg) -> ps.Domain:
    """Build Si substrate with SiO2 pad oxide.

    Stack (bottom to top):
      Si  : y ∈ [-substrate_depth, 0]
      SiO2: y ∈ [0, pad_oxide]          ← pad oxide (screen for implant)
      Air : y ∈ [pad_oxide, pad_oxide + top_space]
    """
    x_extent = cfg["xExtent"]
    substrate_depth = cfg["substrateDepth"]
    top_space = cfg["topSpace"]
    pad_oxide = cfg["padOxideThickness"]
    grid_delta = cfg["gridDelta"]

    domain_top = top_space + pad_oxide
    bounds = [-0.5 * x_extent, 0.5 * x_extent, -substrate_depth, domain_top]
    bc = [
        ps.BoundaryType.REFLECTIVE_BOUNDARY,
        ps.BoundaryType.INFINITE_BOUNDARY,
    ]

    domain = ps.Domain(bounds, bc, grid_delta)

    def _ls():
        return ps.LevelSet(bounds, bc, grid_delta)

    # Si bottom half-space
    ls = _ls()
    ps.MakeGeometry(ls, ps.Plane([0.0, -substrate_depth], [0.0, 1.0])).apply()
    domain.insertNextLevelSetAsMaterial(ls, ps.Material.Si)

    # Si surface at y = 0
    ls = _ls()
    ps.MakeGeometry(ls, ps.Plane([0.0, 0.0], [0.0, 1.0])).apply()
    domain.insertNextLevelSetAsMaterial(ls, ps.Material.Si)

    # SiO2 pad oxide top at y = pad_oxide
    ls = _ls()
    ps.MakeGeometry(ls, ps.Plane([0.0, pad_oxide], [0.0, 1.0])).apply()
    domain.insertNextLevelSetAsMaterial(ls, ps.Material.SiO2)

    domain.generateCellSet(domain_top, ps.Material.Air, True)
    domain.getCellSet().buildNeighborhood()
    return domain


# ── Lateral masking ────────────────────────────────────────────────────────────
_AIR_ID = float(int(ps.Material.Air))
_MASK_ID = float(int(ps.Material.Mask))


def mask_half(domain: ps.Domain, block_right: bool) -> list:
    """
    Convert Air cells on one lateral half to Mask material.

    block_right=True  → mask x > 0  (P implant: keeps left side open)
    block_right=False → mask x < 0  (B implant: keeps right side open)

    Returns the original Material list for later restoration.
    """
    cs = domain.getCellSet()
    mat = list(cs.getScalarData("Material"))
    original = mat[:]

    for i in range(cs.getNumberOfCells()):
        if mat[i] == _AIR_ID:
            x = cs.getCellCenter(i)[0]
            if (block_right and x > 0.0) or (not block_right and x < 0.0):
                mat[i] = _MASK_ID

    cs.setScalarData("Material", mat)
    return original


def restore_material(domain: ps.Domain, original: list):
    domain.getCellSet().setScalarData("Material", original)


# ── Ion implantation ───────────────────────────────────────────────────────────
def run_implant(
    domain,
    conc_label,
    damage_label,
    rp,
    sigma,
    skewness,
    kurtosis,
    rp_tail,
    sigma_tail,
    skewness_tail,
    kurtosis_tail,
    head_fraction,
    dose,
    tilt_deg,
    lateral_sigma_head,
    lateral_sigma_tail,
):
    """Run a dual-Pearson IV IonImplantation step."""
    head = ps.PearsonIVParameters()
    head.mu = rp
    head.sigma = sigma
    head.beta = skewness
    head.gamma = kurtosis

    tail = ps.PearsonIVParameters()
    tail.mu = rp_tail
    tail.sigma = sigma_tail
    tail.beta = skewness_tail
    tail.gamma = kurtosis_tail

    model = ps.ImplantDualPearsonIV(
        head,
        tail,
        head_fraction,
        0.0,
        lateral_sigma_head,
        0.0,
        lateral_sigma_tail,
    )

    implant = ps.IonImplantation()
    implant.setImplantModel(model)
    implant.setDose(dose)
    implant.setTiltAngle(tilt_deg)
    implant.setLengthUnit(1e-7)
    implant.setDoseControl(ps.ImplantDoseControl.WaferDose)
    implant.setMaskMaterials([ps.Material.Mask])
    implant.setScreenMaterials([ps.Material.SiO2])
    implant.setConcentrationLabel(conc_label)
    implant.setDamageLabel(damage_label)

    ps.Process(domain, implant, 0.0).apply()


# ── Anneal helper ──────────────────────────────────────────────────────────────
def make_anneal(
    species_label, active_label, temp_k, duration_s, D0, Ea, sol_C0, sol_Ea
):
    """Configure an Anneal model with Arrhenius diffusivity and solid solubility."""
    anneal = ps.Anneal()
    anneal.setTemperature(temp_k)
    anneal.setSpeciesLabel(species_label)
    anneal.setActiveLabel(active_label)
    anneal.setDiffusionMaterials([ps.Material.Si])
    anneal.setBlockingMaterials([ps.Material.Air, ps.Material.SiO2])
    anneal.setArrheniusParameters(D0, Ea)
    anneal.setDuration(duration_s)
    anneal.enableSolidActivation(True)
    anneal.setSolidSolubilityArrhenius(sol_C0, sol_Ea)
    return anneal


# ── Profile extraction ─────────────────────────────────────────────────────────
def extract_depth_profile(domain: ps.Domain, label):
    """Return (depths_nm, peak_values) — peak concentration per depth slice."""
    cs = domain.getCellSet()
    delta = cs.getGridDelta()
    data = cs.getScalarData(label)
    if data is None:
        return [], []

    bins = {}
    for i in range(cs.getNumberOfCells()):
        v = data[i]
        if v <= 0:
            continue
        depth = -cs.getCellCenter(i)[1]
        if depth < 0:
            continue
        key = round(depth / delta) * delta
        bins[key] = max(bins.get(key, 0.0), v)

    depths = sorted(bins)
    return depths, [bins[d] for d in depths]


def extract_lateral_profile(domain: ps.Domain, label, at_depth):
    """Return (x_coords_nm, values) averaged at positive substrate depth."""
    cs = domain.getCellSet()
    delta = cs.getGridDelta()
    data = cs.getScalarData(label)
    if data is None:
        return [], []

    depth_key = round(at_depth / delta) * delta
    bins = {}
    for i in range(cs.getNumberOfCells()):
        depth = -cs.getCellCenter(i)[1]
        if depth < 0:
            continue
        if abs(round(depth / delta) * delta - depth_key) > delta * 0.5:
            continue
        x = cs.getCellCenter(i)[0]
        key = round(x / delta) * delta
        entry = bins.setdefault(key, [0.0, 0])
        entry[0] += data[i]
        entry[1] += 1

    xs = sorted(bins)
    return xs, [bins[x][0] / bins[x][1] for x in xs]


# ── CSV writer ─────────────────────────────────────────────────────────────────
def write_csv(filename, header, rows):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {filename}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "config",
        nargs="?",
        default="config.txt",
        help="Path to config file (default: config.txt)",
    )
    ap.add_argument("--no-vtk", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg_path = args.config
    if not Path(cfg_path).exists():
        sys.exit(
            f"Config not found: {cfg_path}\n"
            f"Usage: python pnJunction.py [config.txt]"
        )

    cfg = ps.readConfigFile(cfg_path)
    print(f"=== Lateral PN junction (config: {cfg_path}) ===\n")

    # ── Read all parameters from config ───────────────────────────────────────
    grid_delta = cfg["gridDelta"]
    x_extent = cfg["xExtent"]
    substrate_depth = cfg["substrateDepth"]
    anneal_temp_k = cfg["annealTemperatureC"] + 273.15
    anneal_time_s = cfg["annealTimeS"]
    scan_depth = abs(cfg["junctionScanDepthNm"])

    p_dose = cfg["pDoseCm2"]
    p_tilt = cfg["pTiltDeg"]
    p_rot = cfg["pRotationDeg"]
    p_rp = cfg["pProjectedRange"]
    p_sigma = cfg["pDepthSigma"]
    p_skew = cfg["pSkewness"]
    p_kurt = cfg["pKurtosis"]
    p_lsh = cfg["pLateralSigmaHead"]
    p_hf = cfg["pHeadFraction"]
    p_rpt = cfg["pTailProjectedRange"]
    p_sigt = cfg["pTailDepthSigma"]
    p_skewt = cfg["pTailSkewness"]
    p_kurtt = cfg["pTailKurtosis"]
    p_lst = cfg["pTailLateralSigma"]
    p_D0 = cfg["pAnnealD0"]
    p_Ea = cfg["pAnnealEa"]
    p_solC0 = cfg["pSolidSolubilityC0"]
    p_solEa = cfg["pSolidSolubilityEa"]

    b_dose = cfg["bDoseCm2"]
    b_tilt = cfg["bTiltDeg"]
    b_rot = cfg["bRotationDeg"]
    b_rp = cfg["bProjectedRange"]
    b_sigma = cfg["bDepthSigma"]
    b_skew = cfg["bSkewness"]
    b_kurt = cfg["bKurtosis"]
    b_lsh = cfg["bLateralSigmaHead"]
    b_hf = cfg["bHeadFraction"]
    b_rpt = cfg["bTailProjectedRange"]
    b_sigt = cfg["bTailDepthSigma"]
    b_skewt = cfg["bTailSkewness"]
    b_kurtt = cfg["bTailKurtosis"]
    b_lst = cfg["bTailLateralSigma"]
    b_D0 = cfg["bAnnealD0"]
    b_Ea = cfg["bAnnealEa"]
    b_solC0 = cfg["bSolidSolubilityC0"]
    b_solEa = cfg["bSolidSolubilityEa"]

    # ── Build substrate ───────────────────────────────────────────────────────
    domain = build_substrate(cfg)
    print(
        f"Domain: {x_extent} nm wide × {substrate_depth} nm deep,  "
        f"Δ = {grid_delta} nm\n"
    )

    # ── Step 1: P implant — mask right half (x > 0) ───────────────────────────
    print(
        f"Step 1: P implant  {p_dose:.0e} cm⁻²  tilt={p_tilt}°  rot={p_rot}°  (left half, x < 0)"
    )
    orig = mask_half(domain, block_right=True)
    run_implant(
        domain,
        "P_total",
        "P_damage",
        p_rp,
        p_sigma,
        p_skew,
        p_kurt,
        p_rpt,
        p_sigt,
        p_skewt,
        p_kurtt,
        p_hf,
        p_dose,
        p_tilt,
        p_lsh,
        p_lst,
    )
    restore_material(domain, orig)
    print("  P_total written to cell set.\n")

    # ── Step 2: B implant — mask left half (x < 0) ────────────────────────────
    print(
        f"Step 2: B implant  {b_dose:.0e} cm⁻²  tilt={b_tilt}°  rot={b_rot}°  (right half, x > 0)"
    )
    orig = mask_half(domain, block_right=False)
    run_implant(
        domain,
        "B_total",
        "B_damage",
        b_rp,
        b_sigma,
        b_skew,
        b_kurt,
        b_rpt,
        b_sigt,
        b_skewt,
        b_kurtt,
        b_hf,
        b_dose,
        b_tilt,
        b_lsh,
        b_lst,
    )
    restore_material(domain, orig)
    print("  B_total written to cell set.\n")

    # ── Step 3: Zero-time activation ──────────────────────────────────────────
    print("Step 3: Solid activation (no diffusion — initialises P_active, B_active)")
    anneal_P = make_anneal(
        "P_total",
        "P_active",
        anneal_temp_k,
        anneal_time_s,
        p_D0,
        p_Ea,
        p_solC0,
        p_solEa,
    )
    anneal_P.applyActivation(domain)

    anneal_B = make_anneal(
        "B_total",
        "B_active",
        anneal_temp_k,
        anneal_time_s,
        b_D0,
        b_Ea,
        b_solC0,
        b_solEa,
    )
    anneal_B.applyActivation(domain)
    print("  P_active and B_active initialised.\n")

    # ── Step 4: Thermal anneal ────────────────────────────────────────────────
    print(f"Step 4: Anneal  {anneal_temp_k - 273.15:.0f} °C / {anneal_time_s:.0f} s")
    ps.Process(domain, anneal_P, 0.0).apply()
    ps.Process(domain, anneal_B, 0.0).apply()
    print("  Anneal done.\n")

    # ── Step 5: NetDoping — find the lateral junction ─────────────────────────
    print("Step 5: NetDoping analysis")
    nd = ps.NetDoping()
    nd.setCellSet(domain.getCellSet())
    nd.addDonorLabel("P_active")
    nd.addAcceptorLabel("B_active")
    nd.apply()

    xj = nd.lateralJunctionPosition(scan_depth)
    xjs = nd.lateralJunctionPositions(scan_depth)
    if math.isinf(xj):
        print(f"  No lateral junction found at depth {scan_depth:.0f} nm.")
    else:
        print(
            f"  Lateral junction at depth {scan_depth:.0f} nm:  "
            f"x_j = {xj:.1f} nm  ({len(xjs)} crossing(s))"
        )
    print()

    # ── Step 6: Sheet resistance ───────────────────────────────────────────────
    print("Step 6: Sheet resistance  (whole-domain integrals)")
    sr = ps.SheetResistance()
    sr.setCellSet(domain.getCellSet())

    sr.setConcentrationLabel("P_active")
    rsh_n = sr.computeElectron()
    print(f"  Rsh(n-side, P_active) = {rsh_n:.0f} Ω/□")

    sr.setConcentrationLabel("B_active")
    rsh_p = sr.computeHole()
    print(f"  Rsh(p-side, B_active) = {rsh_p:.0f} Ω/□")
    print("  (Each integral spans the whole domain; each side contributes ~half.)\n")

    # ── Step 7: Write output files ─────────────────────────────────────────────
    print("Step 7: Writing output files")

    for label, fname in [
        ("P_active", "pnJunction_P_depth.csv"),
        ("B_active", "pnJunction_B_depth.csv"),
        ("net_doping", "pnJunction_netdoping_depth.csv"),
    ]:
        depths, vals = extract_depth_profile(domain, label)
        write_csv(
            fname,
            ["depth_nm", "value"],
            zip([f"{d:.2f}" for d in depths], [f"{v:.6e}" for v in vals]),
        )

    xs, vals = extract_lateral_profile(domain, "net_doping", scan_depth)
    write_csv(
        "pnJunction_lateral.csv",
        ["x_nm", "net_doping"],
        zip([f"{x:.2f}" for x in xs], [f"{v:.6e}" for v in vals]),
    )

    if not args.no_vtk:
        domain.getCellSet().writeVTU("pnJunction_cellset.vtu")
        print("  Wrote pnJunction_cellset.vtu")

    if args.plot:
        _plot(domain, xs, vals, scan_depth)


def _plot(domain, xs_lat, vals_lat, scan_depth):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    for label, color, name in [
        ("P_active", "blue", "P (n-type)"),
        ("B_active", "red", "B (p-type)"),
        ("net_doping", "green", "Net doping"),
    ]:
        d, v = extract_depth_profile(domain, label)
        ax.semilogy(d, [abs(vi) for vi in v], label=name, color=color)
    ax.set_xlabel("Depth (nm)")
    ax.set_ylabel("Concentration (nm⁻³)")
    ax.set_title("Depth profiles (peak across x)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ax = axes[1]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.plot(xs_lat, vals_lat, "o-", color="green", ms=3)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Net doping (nm⁻³)")
    ax.set_title(f"Lateral profile at depth = {scan_depth:.0f} nm")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[2]
    cs = domain.getCellSet()
    nd_data = cs.getScalarData("net_doping")
    if nd_data:
        xs2d, ys2d, zs2d = [], [], []
        for i in range(cs.getNumberOfCells()):
            c = cs.getCellCenter(i)
            v = nd_data[i]
            if v != 0:
                xs2d.append(c[0])
                ys2d.append(-c[1])
                zs2d.append(v)
        vmax = max(abs(min(zs2d)), abs(max(zs2d)))
        sc = ax.scatter(xs2d, ys2d, c=zs2d, cmap="RdBu_r", s=1, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Net doping (nm⁻³)")
        ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Depth (nm)")
    ax.set_title("Net doping map (blue=n, red=p)")

    plt.suptitle("Lateral PN junction — P (30 keV) | B (20 keV) — 1000 °C/30 s", y=1.01)
    plt.tight_layout()
    plt.savefig("pnJunction.png", dpi=150, bbox_inches="tight")
    print("  Wrote pnJunction.png")
    plt.show()


if __name__ == "__main__":
    main()
