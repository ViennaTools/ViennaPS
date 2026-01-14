import viennaps as ps
import matplotlib.pyplot as plt


def _enum_name(value) -> str:
    return getattr(value, "name", str(value))


def main():

    rows = []
    swatch_colors = []

    for mat in ps.Material:
        info = ps.MaterialInfo(mat)
        rows.append(
            [
                "",
                info.name,
                _enum_name(info.category),
                f"{info.density_gcm3:g}",
                "yes" if info.conductive else "no",
                info.color_rgb,
            ]
        )
        swatch_colors.append(info.color_rgb)

    # Dynamic sizing so the table is readable even with many materials.
    fig_height = max(4.0, 0.35 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")

    col_labels = ["", "Material", "Category", "Density (g/cm³)", "Conductive", "Hex"]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    # Color swatch column: paint the cell background.
    for row_idx, color in enumerate(swatch_colors, start=1):
        cell = table[(row_idx, 0)]
        cell.get_text().set_text("")
        cell.set_facecolor(color)
        cell.set_edgecolor("black")

    # Slightly emphasize header.
    for col_idx in range(len(col_labels)):
        header = table[(0, col_idx)]
        header.set_text_props(weight="bold")

    fig.tight_layout()
    plt.savefig("material_color_table.png", dpi=300)


if __name__ == "__main__":
    main()
