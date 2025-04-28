import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize2d(rates_file, offset, x_extent, trench_width):
    data = np.loadtxt(rates_file, delimiter=",", skiprows=1)
    x = data[:, 0]
    rate = data[:, 1]

    domain_start = offset - x_extent / 2
    domain_end = offset + x_extent / 2
    trench_start = offset - trench_width / 2
    trench_end = offset + trench_width / 2

    plt.figure(figsize=(10, 5))
    plt.plot(x, rate, label="Deposition Rate", linewidth=2)
    plt.axvspan(domain_start, domain_end, color="lightblue", alpha=0.3, label="Simulation Domain")
    plt.axvspan(trench_start, trench_end, color="orange", alpha=0.4, label="Trench Area")
    plt.axvline(offset, color="black", linestyle="--", alpha=0.5, label="Offset X")
    plt.xlabel("x [μm]")
    plt.ylabel("Deposition Rate")
    plt.title("Deposition Rate Profile with Simulation Domain")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("domain2d.png", dpi=300)
    print("Saved rate profile plot as 'domain2d.png'")


def visualize3d(rates_file, offset_x, offset_y, radius, x_extent, y_extent):
    data = np.loadtxt(rates_file, delimiter=",", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    rate = data[:, 2]

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=rate, cmap="coolwarm", s=10)
    plt.colorbar(sc, label="Deposition Rate")

    plt.gca().add_patch(
        plt.Rectangle(
            (offset_x - x_extent / 2, offset_y - y_extent / 2),
            x_extent,
            y_extent,
            fill=False,
            edgecolor="blue",
            linestyle="--",
            linewidth=1.5,
            label="Simulation Domain"
        )
    )
    hole = plt.Circle(
        (offset_x, offset_y),
        radius,
        color="orange",
        alpha=0.3,
        label="Hole Area"
    )
    plt.gca().add_patch(hole)

    plt.xlabel("x [μm]")
    plt.ylabel("y [μm]")
    plt.title("Deposition Rate Map with Simulation Domain")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("domain3d.png", dpi=300)
    print("Saved 3D rate profile plot as 'domain3d.png'")

def parse_config(filename):
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()
    return config


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualization.py <config.txt>")
        sys.exit(1)

    config = parse_config(sys.argv[1])

    try:
        if "offsetY" in config:
            # 3D
            visualize3d(
                rates_file=config["ratesFile"],
                offset_x=float(config["offsetX"]),
                offset_y=float(config["offsetY"]),
                radius=float(config["holeRadius"]),
                x_extent=float(config["xExtent"]),
                y_extent=float(config["yExtent"]),
            )
        else:
            # 2D
            visualize2d(
                rates_file=config["ratesFile"],
                offset=float(config["offsetX"]),
                x_extent=float(config["xExtent"]),
                trench_width=float(config["trenchWidth"]),
            )
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)