#!/usr/bin/env python3
"""Drob et al. 2025 Figure 4 comparison — full-wafer validation runs.

Reads the ps_fw{unc,n15,n30,n45}_step_NNN.vtp surface meshes produced by
locosOxidation.py with the config_fw*.txt configs and reports oxide THICKNESS
against the paper's Deal-Grove model lines, plus the mask film thickness
(volume-preservation check).

Usage (from this directory):
    python3 fig4_table.py            # table only
    python3 fig4_table.py fig4.png   # table + comparison figure

Requires: vtk, numpy (matplotlib additionally for the figure).
Interface material ids in the surface meshes: 10 = Si/oxide; masked runs:
16 = oxide/mask, 30 = mask/gas; uncoated: 30 = oxide/gas.
"""
import glob
import math
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Paper's Deal-Grove constants (dry O2, 1000 C) and native-oxide seed.
B, A, SEED = 0.064, 1.135, 0.0018  # um^2/hr, um, um
GRID = 0.0006                      # um; only used for profile binning

# Experimental points digitized from the paper's Fig. 4 (VASE thickness, nm).
# The paper has no 3 h measurement.  Digitization uncertainty ~ a few nm.
EXPERIMENT = {
    "unc": {1: 84, 2: 107, 4: 198, 5: 223, 6: 275},
    "n15": {1: 56, 2: 97, 4: 187, 5: 217, 6: 265},
    "n30": {1: 5, 2: 15, 4: 29, 5: 44, 6: 42},
    "n45": {1: 2, 2: 7, 4: 5, 5: 10, 6: 13},
}

# tag, label, gamma = D_SiO2/D_mask, e_c (um), oxide-top material id
CURVES = (("unc", "uncoated", 0, 0.0, 30),
          ("n15", "15 cycles", 20, 0.0029, 16),
          ("n30", "30 cycles", 800, 0.0057, 16),
          ("n45", "45 cycles", 1500, 0.0086, 16))


def model(t, Ac):
    """Deal-Grove thickness (nm) at time t for effective constant Ac."""
    tau = SEED * (SEED + Ac) / B
    return (math.sqrt(Ac * Ac / 4 + B * (t + tau)) - Ac / 2) * 1000


def contours(path):
    """Material id -> (N,2) point array from a 2-D surface mesh."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData())[:, :2]
    materials = vtk_to_numpy(grid.GetCellData().GetArray("MaterialIds"))
    out = {}
    for m in np.unique(materials):
        ids = set()
        for i in range(grid.GetNumberOfCells()):
            if materials[i] != m:
                continue
            cell = grid.GetCell(i)
            for j in range(cell.GetNumberOfPoints()):
                ids.add(cell.GetPointId(j))
        out[int(m)] = points[sorted(ids)]
    return out


def surface(pts):
    """Mean upper-surface height (nm) across the domain width."""
    x = pts[:, 0]
    nb = max(8, int(round((x.max() - x.min()) / GRID)))
    xs = np.linspace(x.min(), x.max(), nb + 1)
    heights = [pts[(x >= xs[i]) & (x < xs[i + 1]), 1].max() * 1000
               for i in range(nb) if ((x >= xs[i]) & (x < xs[i + 1])).sum()]
    return float(np.mean(heights))


def collect():
    """{tag: {t: (sim_nm, model_nm, film_nm|None)}} for files present."""
    data = {}
    for tag, _label, g, ec, m_ox in CURVES:
        series = {}
        for t in range(1, 7):
            f = f"ps_fw{tag}_step_{t:03d}.vtp"
            if not glob.glob(f):
                continue
            c = contours(f)
            if 10 not in c or m_ox not in c:
                continue
            sim = surface(c[m_ox]) - surface(c[10])
            film = (surface(c[30]) - surface(c[16])
                    if tag != "unc" and 30 in c and 16 in c else None)
            series[t] = (sim, model(t, A + 2 * g * ec), film)
        data[tag] = series
    return data


def print_table(data):
    print(f"{'t(h)':>4} {'curve':<6} {'SIM':>8} {'MODEL':>8} {'ratio':>7} {'film':>6}")
    print("-" * 44)
    for t in range(1, 7):
        printed = False
        for tag, _label, _g, _ec, _m in CURVES:
            if t not in data[tag]:
                continue
            sim, mod, film = data[tag][t]
            film_s = f"{film:6.2f}" if film is not None else "     -"
            print(f"{t:>4} {tag:<6} {sim:8.2f} {mod:8.2f} {sim / mod:6.3f}x {film_s}")
            printed = True
        if printed:
            print()


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"unc": "#1f4e79", "n15": "#2e7d32", "n30": "#c62828", "n45": "#f9a825"}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    tt = np.linspace(0.05, 6.1, 300)
    for tag, label, g, ec, _m in CURVES:
        Ac = A + 2 * g * ec
        ax.plot(tt, [model(t, Ac) for t in tt], color=colors[tag], lw=1.6)
        ts = sorted(data[tag])
        if ts:
            ax.plot(ts, [data[tag][t][0] for t in ts], "o", ms=6,
                    color=colors[tag], mec="black", mew=0.5)
        exp = EXPERIMENT.get(tag, {})
        if exp:
            ax.plot(sorted(exp), [exp[t] for t in sorted(exp)], "s", ms=7,
                    mfc="none", mec=colors[tag], mew=1.4)
    ax.set_xlabel("oxidation time (h)")
    ax.set_ylabel("SiO$_2$ thickness (nm)")
    ax.set_title("Thermal oxidation with ALD Al$_2$O$_3$ barrier "
                 "(Drob et al. 2025, Fig. 4)")
    ax.set_xlim(0, 6.3)
    ax.set_ylim(0, 290)
    ax.grid(alpha=0.3)

    # Two legends: WHAT each color means, and WHAT each marker style means.
    from matplotlib.lines import Line2D
    coating = [Line2D([], [], color=colors[tag], lw=2.5, label=label)
               for tag, label, _g, _ec, _m in CURVES]
    style = [
        Line2D([], [], color="0.25", lw=1.6,
               label="Deal-Grove model (Drob et al. 2025)"),
        Line2D([], [], color="0.25", marker="s", ls="none", ms=7,
               mfc="none", mew=1.4, label="experiment (Drob et al. 2025)"),
        Line2D([], [], color="0.25", marker="o", ls="none", ms=6,
               mec="black", mew=0.5, label="ViennaPS simulation"),
    ]
    first = ax.legend(handles=coating, title="coating", fontsize=8,
                      title_fontsize=8, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=style, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    results = collect()
    print_table(results)
    if len(sys.argv) > 1:
        plot(results, sys.argv[1])
