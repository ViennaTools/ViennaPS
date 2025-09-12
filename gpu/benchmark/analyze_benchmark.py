#!/usr/bin/env python3
"""
ViennaPS Benchmark Results Analysis Script

This script analyzes the GPU and CPU benchmark results from the ViennaPS simulation,
comparing performance across different sticking coefficients and processing stages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys


class BenchmarkAnalyzer:
    def __init__(self, benchmark_dir="build/gpu/benchmark"):
        """Initialize the benchmark analyzer with the directory containing results."""
        self.benchmark_dir = Path(benchmark_dir)
        self.gpu_data = None
        self.cpu_data = None

    def load_data(self):
        """Load GPU and CPU benchmark data from text files."""
        try:
            # Load GPU benchmark data
            gpu_file = self.benchmark_dir / "GPU_Benchmark.txt"
            if gpu_file.exists():
                self.gpu_data = pd.read_csv(gpu_file, sep=";")
                print(f"Loaded GPU benchmark data: {len(self.gpu_data)} rows")
            else:
                print(f"Warning: GPU benchmark file not found at {gpu_file}")

            # Load CPU benchmark data
            cpu_file = self.benchmark_dir / "CPU_Benchmark.txt"
            if cpu_file.exists():
                self.cpu_data = pd.read_csv(cpu_file, sep=";")
                print(f"Loaded CPU benchmark data: {len(self.cpu_data)} rows")
            else:
                print(f"Warning: CPU benchmark file not found at {cpu_file}")

        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def convert_to_milliseconds(self, data):
        """Convert timing data from nanoseconds to milliseconds for better readability."""
        timing_columns = ["Meshing", "Tracing", "Postprocessing", "Advection"]
        data_ms = data.copy()
        for col in timing_columns:
            if col in data_ms.columns:
                data_ms[col] = data_ms[col] / 1e6  # Convert ns to ms
        return data_ms

    def calculate_statistics(self, data, name):
        """Calculate statistics for each sticking coefficient and processing stage."""
        print(f"\n=== {name} Performance Statistics ===")

        # Convert to milliseconds for analysis
        data_ms = self.convert_to_milliseconds(data)

        # Group by sticking coefficient and calculate statistics
        stats = (
            data_ms.groupby("Sticking")
            .agg(
                {
                    "Meshing": ["mean", "std", "min", "max"],
                    "Tracing": ["mean", "std", "min", "max"],
                    "Postprocessing": ["mean", "std", "min", "max"],
                    "Advection": ["mean", "std", "min", "max"],
                }
            )
            .round(2)
        )

        print(stats)

        # Calculate total time per iteration
        data_ms["Total"] = data_ms[
            ["Meshing", "Tracing", "Postprocessing", "Advection"]
        ].sum(axis=1)
        total_stats = (
            data_ms.groupby("Sticking")["Total"]
            .agg(["mean", "std", "min", "max"])
            .round(2)
        )
        print(f"\nTotal Time Statistics (ms):")
        print(total_stats)

        return data_ms, stats

    def plot_performance_comparison(self, gpu_data_ms, cpu_data_ms):
        """Create comprehensive performance comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "ViennaPS GPU vs CPU Performance Comparison", fontsize=16, fontweight="bold"
        )

        stages = ["Meshing", "Tracing", "Postprocessing", "Advection"]

        # Calculate mean times for each sticking coefficient
        gpu_means = gpu_data_ms.groupby("Sticking")[stages].mean()
        cpu_means = cpu_data_ms.groupby("Sticking")[stages].mean()

        # Plot individual stages
        for i, stage in enumerate(stages):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            sticking_values = gpu_means.index

            ax.plot(
                sticking_values,
                gpu_means[stage],
                "o-",
                label="GPU",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                sticking_values,
                cpu_means[stage],
                "s-",
                label="CPU",
                linewidth=2,
                markersize=6,
            )

            ax.set_xlabel("Sticking Coefficient")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"{stage} Performance", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

        # Plot total time comparison
        ax = axes[1, 2]
        gpu_total = gpu_means.sum(axis=1)
        cpu_total = cpu_means.sum(axis=1)

        ax.plot(
            sticking_values, gpu_total, "o-", label="GPU", linewidth=2, markersize=6
        )
        ax.plot(
            sticking_values, cpu_total, "s-", label="CPU", linewidth=2, markersize=6
        )

        ax.set_xlabel("Sticking Coefficient")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Total Time Performance", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        return fig

    def plot_speedup_analysis(self, gpu_data_ms, cpu_data_ms):
        """Create speedup analysis plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("GPU Speedup Analysis", fontsize=16, fontweight="bold")

        stages = ["Meshing", "Tracing", "Postprocessing", "Advection"]

        # Calculate mean times for speedup calculation
        gpu_means = gpu_data_ms.groupby("Sticking")[stages].mean()
        cpu_means = cpu_data_ms.groupby("Sticking")[stages].mean()

        # Calculate speedup factors
        speedup = cpu_means / gpu_means

        # Plot speedup by stage
        ax = axes[0]
        sticking_values = speedup.index

        for stage in stages:
            ax.plot(
                sticking_values,
                speedup[stage],
                "o-",
                label=stage,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Sticking Coefficient")
        ax.set_ylabel("Speedup Factor (CPU time / GPU time)")
        ax.set_title("Speedup by Processing Stage", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Plot overall speedup
        ax = axes[1]
        total_speedup = cpu_means.sum(axis=1) / gpu_means.sum(axis=1)

        ax.plot(
            sticking_values,
            total_speedup,
            "ro-",
            linewidth=3,
            markersize=8,
            label="Overall Speedup",
        )
        ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="No Speedup")

        ax.set_xlabel("Sticking Coefficient")
        ax.set_ylabel("Speedup Factor")
        ax.set_title("Overall GPU Speedup", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add speedup values as text annotations
        for x, y in zip(sticking_values, total_speedup):
            ax.annotate(
                f"{y:.1f}x",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def plot_performance_breakdown(self, gpu_data_ms, cpu_data_ms):
        """Create performance breakdown pie charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Performance Breakdown by Processing Stage", fontsize=16, fontweight="bold"
        )

        stages = ["Meshing", "Tracing", "Postprocessing", "Advection"]
        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))

        # GPU breakdown for low and high sticking coefficients
        gpu_low = gpu_data_ms[gpu_data_ms["Sticking"] == 0.1][stages].mean()
        gpu_high = gpu_data_ms[gpu_data_ms["Sticking"] == 1.0][stages].mean()

        axes[0, 0].pie(
            gpu_low, labels=stages, autopct="%1.1f%%", colors=colors, startangle=90
        )
        axes[0, 0].set_title("GPU - Low Sticking (0.1)", fontweight="bold")

        axes[0, 1].pie(
            gpu_high, labels=stages, autopct="%1.1f%%", colors=colors, startangle=90
        )
        axes[0, 1].set_title("GPU - High Sticking (1.0)", fontweight="bold")

        # CPU breakdown for low and high sticking coefficients
        cpu_low = cpu_data_ms[cpu_data_ms["Sticking"] == 0.1][stages].mean()
        cpu_high = cpu_data_ms[cpu_data_ms["Sticking"] == 1.0][stages].mean()

        axes[1, 0].pie(
            cpu_low, labels=stages, autopct="%1.1f%%", colors=colors, startangle=90
        )
        axes[1, 0].set_title("CPU - Low Sticking (0.1)", fontweight="bold")

        axes[1, 1].pie(
            cpu_high, labels=stages, autopct="%1.1f%%", colors=colors, startangle=90
        )
        axes[1, 1].set_title("CPU - High Sticking (1.0)", fontweight="bold")

        plt.tight_layout()
        return fig

    def generate_summary_report(self, gpu_data_ms, cpu_data_ms):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("VIENNAPS BENCHMARK ANALYSIS SUMMARY")
        print("=" * 80)

        stages = ["Meshing", "Tracing", "Postprocessing", "Advection"]

        # Overall performance comparison
        gpu_total_mean = gpu_data_ms[stages].sum(axis=1).mean()
        cpu_total_mean = cpu_data_ms[stages].sum(axis=1).mean()
        overall_speedup = cpu_total_mean / gpu_total_mean

        print(f"\nOVERALL PERFORMANCE:")
        print(f"Average GPU total time: {gpu_total_mean:.2f} ms")
        print(f"Average CPU total time: {cpu_total_mean:.2f} ms")
        print(f"Overall GPU speedup: {overall_speedup:.2f}x")

        # Stage-wise analysis
        print(f"\nSTAGE-WISE SPEEDUP ANALYSIS:")
        gpu_stage_means = gpu_data_ms[stages].mean()
        cpu_stage_means = cpu_data_ms[stages].mean()

        for stage in stages:
            speedup = cpu_stage_means[stage] / gpu_stage_means[stage]
            print(f"{stage:15}: {speedup:6.2f}x speedup")

        # Best and worst case scenarios
        gpu_total_by_run = gpu_data_ms[stages].sum(axis=1)
        cpu_total_by_run = cpu_data_ms[stages].sum(axis=1)

        gpu_means_by_sticking = gpu_data_ms.groupby("Sticking").apply(
            lambda x: x[stages].sum(axis=1).mean()
        )
        cpu_means_by_sticking = cpu_data_ms.groupby("Sticking").apply(
            lambda x: x[stages].sum(axis=1).mean()
        )
        speedups_by_sticking = cpu_means_by_sticking / gpu_means_by_sticking

        best_sticking = speedups_by_sticking.idxmax()
        worst_sticking = speedups_by_sticking.idxmin()

        print(f"\nPERFORMANCE BY STICKING COEFFICIENT:")
        print(
            f"Best GPU performance (highest speedup): Sticking = {best_sticking} ({speedups_by_sticking[best_sticking]:.2f}x)"
        )
        print(
            f"Worst GPU performance (lowest speedup): Sticking = {worst_sticking} ({speedups_by_sticking[worst_sticking]:.2f}x)"
        )

        # Performance variability
        gpu_total_by_run = gpu_data_ms[stages].sum(axis=1)
        cpu_total_by_run = cpu_data_ms[stages].sum(axis=1)

        gpu_std = gpu_data_ms.groupby("Sticking").apply(
            lambda x: x[stages].sum(axis=1).std()
        )
        cpu_std = cpu_data_ms.groupby("Sticking").apply(
            lambda x: x[stages].sum(axis=1).std()
        )

        print(f"\nPERFORMANCE VARIABILITY:")
        print(f"GPU standard deviation: {gpu_std.mean():.2f} ms")
        print(f"CPU standard deviation: {cpu_std.mean():.2f} ms")
        print(f"GPU coefficient of variation: {gpu_std.mean()/gpu_total_mean*100:.2f}%")
        print(f"CPU coefficient of variation: {cpu_std.mean()/cpu_total_mean*100:.2f}%")

    def save_processed_data(
        self, gpu_data_ms, cpu_data_ms, output_dir="benchmark_analysis"
    ):
        """Save processed data and summary statistics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save processed data
        gpu_data_ms.to_csv(output_path / "gpu_benchmark_processed.csv", index=False)
        cpu_data_ms.to_csv(output_path / "cpu_benchmark_processed.csv", index=False)

        # Save summary statistics
        stages = ["Meshing", "Tracing", "Postprocessing", "Advection"]

        gpu_stats = gpu_data_ms.groupby("Sticking")[stages].agg(
            ["mean", "std", "min", "max"]
        )
        cpu_stats = cpu_data_ms.groupby("Sticking")[stages].agg(
            ["mean", "std", "min", "max"]
        )

        gpu_stats.to_csv(output_path / "gpu_statistics.csv")
        cpu_stats.to_csv(output_path / "cpu_statistics.csv")

        print(f"\nProcessed data and statistics saved to: {output_path}")

    def run_analysis(self, save_plots=True, save_data=True):
        """Run the complete benchmark analysis."""
        print("Starting ViennaPS Benchmark Analysis...")

        # Load data
        self.load_data()

        if self.gpu_data is None or self.cpu_data is None:
            print("Error: Could not load benchmark data. Please check file paths.")
            return

        # Calculate statistics
        gpu_data_ms, gpu_stats = self.calculate_statistics(self.gpu_data, "GPU")
        cpu_data_ms, cpu_stats = self.calculate_statistics(self.cpu_data, "CPU")

        # Generate plots
        if save_plots:
            plt.style.use(
                "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
            )

            # Performance comparison plot
            fig1 = self.plot_performance_comparison(gpu_data_ms, cpu_data_ms)
            fig1.savefig(
                "benchmark_performance_comparison.png", dpi=300, bbox_inches="tight"
            )

            # Speedup analysis plot
            fig2 = self.plot_speedup_analysis(gpu_data_ms, cpu_data_ms)
            fig2.savefig("benchmark_speedup_analysis.png", dpi=300, bbox_inches="tight")

            # Performance breakdown plot
            fig3 = self.plot_performance_breakdown(gpu_data_ms, cpu_data_ms)
            fig3.savefig(
                "benchmark_performance_breakdown.png", dpi=300, bbox_inches="tight"
            )

            print("\nPlots saved:")
            print("- benchmark_performance_comparison.png")
            print("- benchmark_speedup_analysis.png")
            print("- benchmark_performance_breakdown.png")

            plt.show()

        # Generate summary report
        self.generate_summary_report(gpu_data_ms, cpu_data_ms)

        # Save processed data
        if save_data:
            self.save_processed_data(gpu_data_ms, cpu_data_ms)


def main():
    """Main function to run the benchmark analysis."""
    parser = argparse.ArgumentParser(description="Analyze ViennaPS benchmark results")
    parser.add_argument(
        "--benchmark-dir",
        "-d",
        default="build/gpu/benchmark",
        help="Directory containing benchmark results (default: build/gpu/benchmark)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving processed data"
    )

    args = parser.parse_args()

    analyzer = BenchmarkAnalyzer(args.benchmark_dir)
    analyzer.run_analysis(save_plots=not args.no_plots, save_data=not args.no_save)


if __name__ == "__main__":
    main()
