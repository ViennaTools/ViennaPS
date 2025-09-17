import numpy as np
import matplotlib.pyplot as plt
import os
import glob


# Function to plot final exposure grid
def plot_final_exposure(final_exposure_file):

    if os.path.exists(final_exposure_file):
        exposure_grid = np.loadtxt(final_exposure_file, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.imshow(
            exposure_grid,
            cmap="jet",
            origin="lower",
            alpha=0.5,
            interpolation="nearest",
        )  # Adjust alpha for lighter colors
        plt.colorbar(label="Exposure Intensity")
        plt.title(f"Final Exposure Grid")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()
    else:
        print(f"Final exposure file not found.")


plot_final_exposure("finalGrid.csv")
