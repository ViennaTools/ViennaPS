import argparse
from os import path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def file_type(string):
    if path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


parser = argparse.ArgumentParser(description="Plot interpolation results.")
parser.add_argument(
    "points_file",
    type=file_type,
    help="path to CSV file containing the input points.",
)

parser.add_argument(
    "data_file",
    type=file_type,
    help="path to CSV file containing the interpolated data.",
)


def plot(
    fig: mpl.figure.Figure,  # figure to plot on
    gspec: gridspec.SubplotSpec,  # subplot specification
    points: np.ndarray,  # points to plot, as NumPy array
    data: np.ndarray,  # data to plot, as a NumPy array
    extent: tuple[float, float, float, float],  # extent of the plotted data
) -> tuple[mpl.axes.Axes, mpl.cm.ScalarMappable]:
    """Plots an image of the given data on a subplot of the given figure with the data
    points overlayed as scatter plot.

    Args:
        fig: The figure to plot on.
        gspec: The subplot specification for the figure.
        points: The points to plot, as a NumPy array.
        data: The data to plot, as a NumPy array.
        extent: The extent of the plotted data, as a tuple of floats in the form (xmin,
            xmax, ymin, ymax).

    Returns:
        A tuple containing the Axes object for the subplot and a scalar mappable object
        for the plotted data.
    """
    # create a normalization object for the data
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    # create a colormap object
    cmap = mpl.colormaps["jet"]

    # add a subplot to the figure and assign it to the variable ax
    ax = fig.add_subplot(gspec)

    # set the title and labels for the subplot
    ax.set_title("Predicted value")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")

    # plot the data as an image on the subplot
    ax.imshow(
        data,
        extent=extent,
        interpolation="none",
        origin="lower",
        aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]),
        norm=norm,
        cmap=cmap,
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        1,
        color="r",
    )

    # return the subplot and the scalar mappable object
    return ax, mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def main(args: argparse.Namespace) -> None:
    points = np.loadtxt(args.points_file, delimiter=",")
    if points.shape[1] != 3:
        print("Points are not in the format x,y,data")
        return

    data = np.loadtxt(args.data_file, delimiter=",")
    if data.shape[1] != 3:
        print("Data is not in the format x,y,data")
        return

    # Get unique values for x, y, and z
    xticks = np.unique(data[:, 0])
    yticks = np.unique(data[:, 1])

    # Calculate the extent of the plotted data
    extent = (xticks[0], xticks[-1], yticks[0], yticks[-1])

    # Create a figure with a 2x5 grid of subplots
    fig = plt.figure()

    # Create a GridSpec object for the figure
    gspec = gridspec.GridSpec(1, 8, figure=fig)

    # Plot the value data on the first subplot
    _, sm = plot(
        fig,
        gspec[0, :-1],
        points,
        data[:, 2].reshape(xticks.shape[0], yticks.shape[0]),
        extent=extent,
    )

    # Add a colorbar for the value data
    fig.colorbar(
        sm,
        cax=fig.add_subplot(gspec[0, -1]),
        orientation="vertical",
        label="Predicted Value",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
