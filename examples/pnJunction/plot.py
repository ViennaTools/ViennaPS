import numpy as np
import matplotlib.pyplot as plt

files = {
    "B depth profile": "pnJunction_B_depth.csv",
    "P depth profile": "pnJunction_P_depth.csv",
    "Net depth profile": "pnJunction_net_depth.csv",
    "Lateral profile": "pnJunction_lateral.csv",
}

fig, axes = plt.subplots(
    nrows=len(files),
    ncols=2,
    figsize=(11, 10),
    constrained_layout=True,
)

for row, (title, filename) in enumerate(files.items()):
    # Read header
    with open(filename, "r") as f:
        header = f.readline().strip().split(",")

    # Load numerical data
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Detect x-axis column
    if "depth_nm" in header:
        x_col = header.index("depth_nm")
        x_label = "Depth [nm]"
    elif "x_nm" in header:
        x_col = header.index("x_nm")
        x_label = "x [nm]"
    else:
        raise ValueError(f"Could not find position column in {filename}")

    y_col = header.index("value")

    x = data[:, x_col]
    y = data[:, y_col]

    # Linear plot
    ax_lin = axes[row, 0]
    ax_lin.plot(x, y, linewidth=2)
    ax_lin.set_title(f"{title} — linear")
    ax_lin.set_xlabel(x_label)
    ax_lin.set_ylabel("Value")
    ax_lin.grid(True)

    # Log-y plot
    ax_log = axes[row, 1]
    ax_log.semilogy(x, np.clip(y, 1e-30, None), linewidth=2)
    ax_log.set_title(f"{title} — log y")
    ax_log.set_xlabel(x_label)
    ax_log.set_ylabel("Value")
    ax_log.grid(True, which="both")

plt.show()