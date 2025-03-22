import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Function to load and plot polygons for a given layer
def plot_GDSIIpolygons(layer):
    polygon_files = sorted(glob.glob(f"./{layer}_GDSIIpolygon*.csv"))
    
    if not polygon_files:
        print(f"No polygons found for {layer}.")
        return
    
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "violet", "orange", "cyan", "magenta"]
    
    for i, file in enumerate(polygon_files):
        contour = np.loadtxt(file, delimiter=",")
        contour = np.vstack((contour, contour[0, :]))  # Close the polygon
        plt.plot(contour[:, 0], contour[:, 1], color=colors[i % len(colors)], linestyle="-", label=f"Polygon {i}")
    
    plt.title(f"Extracted GDSII Polygons - {layer}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot the exposure grid
def plot_exposure_grid(layer):
    exposure_file = f"./{layer}_exposure.csv"
    if os.path.exists(exposure_file):
        exposure_grid = np.loadtxt(exposure_file, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.imshow(exposure_grid, cmap="hot", origin="lower", interpolation="nearest")
        plt.colorbar(label="Exposure Location")
        plt.xlabel("X Grid Index")
        plt.ylabel("Y Grid Index")
        plt.title(f"Rasterized Exposure Grid - {layer}")
        plt.show()
    else:
        print(f"Exposure grid file not found for {layer}.")

# Function to plot final exposure grid
def plot_final_exposure(layer):
    final_exposure_file = f"./{layer}_finalExposure.csv"
    contour_file = f"./{layer}_allContours.csv"
    
    if os.path.exists(final_exposure_file):
        exposure_grid = np.loadtxt(final_exposure_file, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.imshow(exposure_grid, cmap="jet", origin="lower", alpha=0.25, interpolation="nearest")  # Adjust alpha for lighter colors
        
        # Overlay extracted contours if available
        if os.path.exists(contour_file):
            contours = np.loadtxt(contour_file, delimiter=",")
            plt.scatter(contours[:, 0], contours[:, 1], color="black", marker = '.', s=4, label="Extracted Contours")
        
        plt.colorbar(label="Exposure Intensity")
        plt.title(f"Final Exposure Grid with Contours - {layer}")
        plt.xlabel("X Coordinate (µm)")
        plt.ylabel("Y Coordinate (µm)")
        plt.legend()
        plt.show()
    else:
        print(f"Final exposure file not found for {layer}.")

# Function to plot extracted contour points
def plot_contours(layer):
    contour_file = f"./{layer}_allContours.csv"
    if os.path.exists(contour_file):
        contours = np.loadtxt(contour_file, delimiter=",")
        plt.figure(figsize=(8, 6))
        plt.scatter(contours[:, 0], contours[:, 1], color="red", marker="o", label="Contour Points")
        plt.title(f"Extracted Single Contour Points - {layer}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()
    else:
        print(f"Contours file not found for {layer}.")

# Function to plot extracted polygons
def plot_polygons(layer):
    polygon_files = sorted(glob.glob(f"./{layer}_polygon*.csv"))
    
    if not polygon_files:
        print(f"No polygons found for {layer}.")
        return
    
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "violet", "orange", "cyan", "magenta"]
    
    for i, file in enumerate(polygon_files):
        contour = np.loadtxt(file, delimiter=",")
        contour = np.vstack((contour, contour[0, :]))  # Close the polygon
        plt.plot(contour[:, 0], contour[:, 1], color=colors[i % len(colors)], linestyle="-", label=f"Polygon {i}")
    
    plt.title(f"Extracted GDSII Polygons - {layer}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot simplified open contours
def plot_simplified_polygons(layer):
    simplified_files = sorted(glob.glob(f"./{layer}_simplePolygon*.csv"))
    
    if not simplified_files:
        print(f"No simplified polygons found for {layer}.")
        return
    
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "violet", "orange", "cyan", "magenta"]
    
    for i, file in enumerate(simplified_files):
        scontour = np.loadtxt(file, delimiter=",")
        plt.plot(scontour[:, 0], scontour[:, 1], color=colors[i % len(colors)], linestyle="-", marker='o', label=f"Contour {i}")
    
    plt.title(f"Simplified Open Contours - {layer}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot extracted polygons and simplified polygons as points
def plot_polygons_with_simplified(layer):
    polygon_files = sorted(glob.glob(f"./{layer}_polygon*.csv"))
    simplified_files = sorted(glob.glob(f"./{layer}_simplePolygon*.csv"))
    
    if not polygon_files and not simplified_files:
        print(f"No polygons or simplified polygons found for {layer}.")
        return
    
    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "violet", "orange", "cyan", "magenta"]

    # Plot full polygons as lines
    if polygon_files:
        for i, file in enumerate(polygon_files):
            contour = np.loadtxt(file, delimiter=",")
            contour = np.vstack((contour, contour[0, :]))  # Close the polygon
            plt.plot(contour[:, 0], contour[:, 1], color=colors[i % len(colors)], linestyle="-", linewidth=1, label=f"Polygon {i}")

    # Plot simplified polygons as points
    if simplified_files:
        for i, file in enumerate(simplified_files):
            scontour = np.loadtxt(file, delimiter=",")
            plt.scatter(scontour[:, 0], scontour[:, 1], color=colors[i % len(colors)], marker='o', s=20)

    # Add labels, title, and grid
    plt.title(f"Full Polygons & Simplified Polygons (Points) - {layer}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Identify all layers dynamically
layer_files = sorted(set(f.split("_")[0] for f in os.listdir(".") if f.startswith("layer")))

# Process each layer in order
for layer in layer_files:
    print(f"Processing {layer}...")
    # plot_GDSIIpolygons(layer)
    plot_exposure_grid(layer)
    plot_final_exposure(layer)
    plot_polygons_with_simplified(layer)
    # plot_contours(layer)
    # plot_polygons(layer)
    # plot_simplified_polygons(layer)

print("All layers processed.")