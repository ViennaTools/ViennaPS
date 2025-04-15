import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_exposure_grid(exposure_file="finalGrid.csv"):
    if os.path.exists(exposure_file):
        exposure_grid = np.loadtxt(exposure_file, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.imshow(exposure_grid, cmap="jet", origin="lower", alpha=0.5, interpolation="nearest")
        plt.colorbar(label="Exposure Intensity")
        plt.title(f"Exposure Grid: {exposure_file}")
        plt.xlabel("X Coordinate (µm)")
        plt.ylabel("Y Coordinate (µm)")
        plt.show()
    else:
        print(f"Exposure file '{exposure_file}' not found.")

if __name__ == "__main__":
    # Use first argument as filename if provided
    if len(sys.argv) > 1:
        plot_exposure_grid(sys.argv[1])
    else:
        plot_exposure_grid()

