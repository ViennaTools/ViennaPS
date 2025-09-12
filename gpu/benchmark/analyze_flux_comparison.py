#!/usr/bin/env python3
"""
ViennaPS Flux Comparison Analysis Script

This script analyzes the flux comparison results from compare_flux.cpp,
comparing GPU vs CPU flux calculations on different sides of the geometry.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


class FluxComparisonAnalyzer:
    def __init__(self, data_dir="build/gpu/benchmark"):
        """Initialize the flux comparison analyzer."""
        self.data_dir = Path(data_dir)
        self.flux_data = {}

    def load_flux_data(self):
        """Load flux comparison data from CSV files."""
        flux_files = {
            "top": "compare_flux_top.txt",
            "bottom": "compare_flux_bottom.txt",
            "side_1": "compare_flux_side_1.txt",
            "side_2": "compare_flux_side_2.txt",
        }

        for location, filename in flux_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    data = {"positions": [], "cpu_flux": [], "gpu_flux": []}

                    with open(filepath, "r") as file:
                        reader = csv.DictReader(file, delimiter=";")
                        for row in reader:
                            if location in ["top", "bottom"]:
                                # For top and bottom, use x-coordinate
                                pos = float(row["x"])
                            else:
                                # For sides, use y-coordinate
                                pos = float(row["y"])

                            data["positions"].append(pos)
                            data["cpu_flux"].append(float(row["CPU"]))
                            data["gpu_flux"].append(float(row["GPU"]))

                    # Convert to numpy arrays and sort by position
                    positions = np.array(data["positions"])
                    cpu_flux = np.array(data["cpu_flux"])
                    gpu_flux = np.array(data["gpu_flux"])

                    # Sort by position
                    sort_idx = np.argsort(positions)
                    self.flux_data[location] = {
                        "positions": positions[sort_idx],
                        "cpu_flux": cpu_flux[sort_idx],
                        "gpu_flux": gpu_flux[sort_idx],
                    }

                    print(f"Loaded {len(positions)} flux data points for {location}")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Warning: {filename} not found in {self.data_dir}")

    def calculate_statistics(self):
        """Calculate comparison statistics for each location."""
        print("\n" + "=" * 60)
        print("FLUX COMPARISON STATISTICS")
        print("=" * 60)

        for location, data in self.flux_data.items():
            cpu_flux = data["cpu_flux"]
            gpu_flux = data["gpu_flux"]

            if len(cpu_flux) == 0:
                print(f"\n{location.upper()} SURFACE:")
                print("  No data points available")
                continue

            # Calculate statistics
            relative_error = np.abs(gpu_flux - cpu_flux) / np.abs(cpu_flux) * 100
            absolute_error = np.abs(gpu_flux - cpu_flux)

            print(f"\n{location.upper()} SURFACE:")
            print(f"  Number of points: {len(cpu_flux)}")
            print(f"  CPU flux range: [{cpu_flux.min():.6f}, {cpu_flux.max():.6f}]")
            print(f"  GPU flux range: [{gpu_flux.min():.6f}, {gpu_flux.max():.6f}]")
            print(f"  Mean relative error: {relative_error.mean():.4f}%")
            print(f"  Max relative error: {relative_error.max():.4f}%")
            print(f"  Mean absolute error: {absolute_error.mean():.8f}")
            print(f"  RMS error: {np.sqrt(np.mean(absolute_error**2)):.8f}")

            # Correlation coefficient
            if len(cpu_flux) > 1:
                correlation = np.corrcoef(cpu_flux, gpu_flux)[0, 1]
                print(f"  Correlation coefficient: {correlation:.6f}")

    def create_flux_comparison_plots(self):
        """Create comprehensive flux comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "GPU vs CPU Flux Comparison on Different Geometry Surfaces",
            fontsize=16,
            fontweight="bold",
        )

        locations = ["top", "bottom", "side_1", "side_2"]
        titles = ["Top Surface", "Bottom Surface", "Side Surface 1", "Side Surface 2"]

        for i, (location, title) in enumerate(zip(locations, titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if (
                location not in self.flux_data
                or len(self.flux_data[location]["cpu_flux"]) == 0
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(title, fontweight="bold")
                continue

            data = self.flux_data[location]
            positions = data["positions"]
            cpu_flux = data["cpu_flux"]
            gpu_flux = data["gpu_flux"]

            # Plot flux values
            ax.plot(
                positions,
                cpu_flux,
                "o-",
                label="CPU",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )
            ax.plot(
                positions,
                gpu_flux,
                "s-",
                label="GPU",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

            ax.set_xlabel(
                "Position" + (" (x)" if location in ["top", "bottom"] else " (y)")
            )
            ax.set_ylabel("Flux")
            ax.set_title(title, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add error information
            relative_error = np.abs(gpu_flux - cpu_flux) / np.abs(cpu_flux) * 100
            ax.text(
                0.02,
                0.98,
                f"Mean rel. error: {relative_error.mean():.2f}%",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            )

        plt.tight_layout()
        return fig

    def create_error_analysis_plots(self):
        """Create error analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Flux Comparison Error Analysis", fontsize=16, fontweight="bold")

        locations = ["top", "bottom", "side_1", "side_2"]
        titles = ["Top Surface", "Bottom Surface", "Side Surface 1", "Side Surface 2"]

        for i, (location, title) in enumerate(zip(locations, titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if (
                location not in self.flux_data
                or len(self.flux_data[location]["cpu_flux"]) == 0
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(f"{title} - Relative Error", fontweight="bold")
                continue

            data = self.flux_data[location]
            positions = data["positions"]
            cpu_flux = data["cpu_flux"]
            gpu_flux = data["gpu_flux"]

            # Calculate relative error
            relative_error = (gpu_flux - cpu_flux) / cpu_flux * 100

            # Plot relative error
            ax.plot(positions, relative_error, "ro-", linewidth=2, markersize=4)
            ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

            ax.set_xlabel(
                "Position" + (" (x)" if location in ["top", "bottom"] else " (y)")
            )
            ax.set_ylabel("Relative Error (%)")
            ax.set_title(f"{title} - Relative Error", fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add statistics
            if len(relative_error) > 0:
                mean_error = relative_error.mean()
                std_error = relative_error.std() if len(relative_error) > 1 else 0.0
                ax.text(
                    0.02,
                    0.98,
                    f"Mean: {mean_error:.3f}%\nStd: {std_error:.3f}%",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7
                    ),
                )

        plt.tight_layout()
        return fig

    def create_scatter_plots(self):
        """Create scatter plots for GPU vs CPU flux comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("GPU vs CPU Flux Scatter Plots", fontsize=16, fontweight="bold")

        locations = ["top", "bottom", "side_1", "side_2"]
        titles = ["Top Surface", "Bottom Surface", "Side Surface 1", "Side Surface 2"]
        colors = ["blue", "red", "green", "orange"]

        for i, (location, title, color) in enumerate(zip(locations, titles, colors)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if (
                location not in self.flux_data
                or len(self.flux_data[location]["cpu_flux"]) == 0
            ):
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(title, fontweight="bold")
                continue

            data = self.flux_data[location]
            cpu_flux = data["cpu_flux"]
            gpu_flux = data["gpu_flux"]

            # Scatter plot
            ax.scatter(cpu_flux, gpu_flux, c=color, alpha=0.6, s=50)

            # Perfect correlation line
            min_val = min(cpu_flux.min(), gpu_flux.min())
            max_val = max(cpu_flux.max(), gpu_flux.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Perfect Correlation",
            )

            ax.set_xlabel("CPU Flux")
            ax.set_ylabel("GPU Flux")
            ax.set_title(title, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add correlation coefficient
            if len(cpu_flux) > 1:
                correlation = np.corrcoef(cpu_flux, gpu_flux)[0, 1]
                ax.text(
                    0.02,
                    0.98,
                    f"R = {correlation:.6f}",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

        plt.tight_layout()
        return fig

    def create_summary_plot(self):
        """Create a summary plot showing all surfaces together."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Flux Comparison Summary", fontsize=16, fontweight="bold")

        # Collect all data for statistical comparison
        all_cpu_flux = []
        all_gpu_flux = []
        all_relative_errors = []
        location_labels = []

        colors = ["blue", "red", "green", "orange"]
        locations = ["top", "bottom", "side_1", "side_2"]
        labels = ["Top", "Bottom", "Side 1", "Side 2"]

        for i, (location, label, color) in enumerate(zip(locations, labels, colors)):
            if (
                location in self.flux_data
                and len(self.flux_data[location]["cpu_flux"]) > 0
            ):
                data = self.flux_data[location]
                cpu_flux = data["cpu_flux"]
                gpu_flux = data["gpu_flux"]
                relative_error = np.abs(gpu_flux - cpu_flux) / np.abs(cpu_flux) * 100

                all_cpu_flux.extend(cpu_flux)
                all_gpu_flux.extend(gpu_flux)
                all_relative_errors.extend(relative_error)
                location_labels.extend([label] * len(cpu_flux))

                # Scatter plot for all surfaces
                ax1.scatter(cpu_flux, gpu_flux, c=color, alpha=0.6, s=30, label=label)

        # Perfect correlation line
        if all_cpu_flux:
            min_val = min(all_cpu_flux)
            max_val = max(all_cpu_flux)
            ax1.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                alpha=0.5,
                label="Perfect Correlation",
            )

            ax1.set_xlabel("CPU Flux")
            ax1.set_ylabel("GPU Flux")
            ax1.set_title("All Surfaces Combined", fontweight="bold")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Overall correlation
            correlation = np.corrcoef(all_cpu_flux, all_gpu_flux)[0, 1]
            ax1.text(
                0.02,
                0.98,
                f"Overall R = {correlation:.6f}",
                transform=ax1.transAxes,
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7),
            )

        # Box plot of relative errors by surface
        error_data = []
        error_labels = []
        for location, label in zip(locations, labels):
            if (
                location in self.flux_data
                and len(self.flux_data[location]["cpu_flux"]) > 0
            ):
                data = self.flux_data[location]
                cpu_flux = data["cpu_flux"]
                gpu_flux = data["gpu_flux"]
                relative_error = np.abs(gpu_flux - cpu_flux) / np.abs(cpu_flux) * 100
                error_data.append(relative_error)
                error_labels.append(label)

        if error_data:
            ax2.boxplot(error_data, labels=error_labels)
            ax2.set_ylabel("Absolute Relative Error (%)")
            ax2.set_title("Error Distribution by Surface", fontweight="bold")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_analysis_data(self, output_dir="flux_analysis"):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save detailed comparison data
        for location, data in self.flux_data.items():
            filename = output_path / f"flux_comparison_{location}.csv"
            with open(filename, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Position",
                        "CPU_Flux",
                        "GPU_Flux",
                        "Absolute_Error",
                        "Relative_Error_Percent",
                    ]
                )

                positions = data["positions"]
                cpu_flux = data["cpu_flux"]
                gpu_flux = data["gpu_flux"]

                for pos, cpu, gpu in zip(positions, cpu_flux, gpu_flux):
                    abs_error = abs(gpu - cpu)
                    rel_error = abs_error / abs(cpu) * 100 if cpu != 0 else 0
                    writer.writerow([pos, cpu, gpu, abs_error, rel_error])

        print(f"\nAnalysis data saved to: {output_path}")

    def run_analysis(self, save_plots=True, save_data=True):
        """Run the complete flux comparison analysis."""
        print("Starting Flux Comparison Analysis...")

        # Load data
        self.load_flux_data()

        if not self.flux_data:
            print("Error: No flux data could be loaded. Please check file paths.")
            return

        # Calculate statistics
        self.calculate_statistics()

        # Generate plots
        if save_plots:
            try:
                # Flux comparison plots
                fig1 = self.create_flux_comparison_plots()
                fig1.savefig("flux_comparison_plots.png", dpi=300, bbox_inches="tight")

                # Error analysis plots
                fig2 = self.create_error_analysis_plots()
                fig2.savefig("flux_error_analysis.png", dpi=300, bbox_inches="tight")

                # Scatter plots
                fig3 = self.create_scatter_plots()
                fig3.savefig("flux_scatter_plots.png", dpi=300, bbox_inches="tight")

                # Summary plot
                fig4 = self.create_summary_plot()
                fig4.savefig("flux_summary_plot.png", dpi=300, bbox_inches="tight")

                print("\nPlots saved:")
                print("- flux_comparison_plots.png")
                print("- flux_error_analysis.png")
                print("- flux_scatter_plots.png")
                print("- flux_summary_plot.png")

                plt.show()

            except ImportError:
                print("Warning: matplotlib not available. Plots not generated.")

        # Save analysis data
        if save_data:
            self.save_analysis_data()


def main():
    """Main function to run the flux comparison analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze ViennaPS flux comparison results"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="build/gpu/benchmark",
        help="Directory containing flux comparison files (default: build/gpu/benchmark)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving analysis data"
    )

    args = parser.parse_args()

    analyzer = FluxComparisonAnalyzer(args.data_dir)
    analyzer.run_analysis(save_plots=not args.no_plots, save_data=not args.no_save)


if __name__ == "__main__":
    main()
