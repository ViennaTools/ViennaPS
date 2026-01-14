#!/usr/bin/env python3
"""
ViennaPS Single Benchmark Results Analysis Script

This script analyzes the GPU and CPU single benchmark results from the ViennaPS simulation,
comparing performance between GPU and CPU implementations for meshing, tracing, and postprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys


class SingleBenchmarkAnalyzer:
    def __init__(self, benchmark_dir="build/gpu/benchmark"):
        """Initialize the single benchmark analyzer with the directory containing results."""
        self.benchmark_dir = Path(benchmark_dir)
        self.gpu_data = None
        self.cpu_data = None

    def load_data(self):
        """Load GPU and CPU single benchmark data from text files."""
        try:
            # Load GPU single benchmark data
            gpu_file = self.benchmark_dir / "GPU_Benchmark_single_no_ray_count.txt"
            if gpu_file.exists():
                self.gpu_data = pd.read_csv(gpu_file, sep=";")
                print(f"Loaded GPU single benchmark data: {len(self.gpu_data)} rows")
                print(f"GPU columns: {list(self.gpu_data.columns)}")
            else:
                print(f"Warning: GPU single benchmark file not found at {gpu_file}")

            # Load CPU single benchmark data
            cpu_file = self.benchmark_dir / "CPU_Benchmark_single.txt"
            if cpu_file.exists():
                self.cpu_data = pd.read_csv(cpu_file, sep=";")
                print(f"Loaded CPU single benchmark data: {len(self.cpu_data)} rows")
                print(f"CPU columns: {list(self.cpu_data.columns)}")
            else:
                print(f"Warning: CPU single benchmark file not found at {cpu_file}")

        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def convert_to_milliseconds(self, data):
        """Convert timing data from nanoseconds to milliseconds for better readability."""
        timing_columns = ["Meshing", "Tracing", "Postprocessing"]
        data_ms = data.copy()
        for col in timing_columns:
            if col in data_ms.columns:
                data_ms[col] = data_ms[col] / 1e6  # Convert ns to ms
        return data_ms

    def calculate_statistics(self, data, name):
        """Calculate statistics for each processing stage."""
        print(f"\n=== {name} Single Benchmark Performance Statistics ===")

        # Convert to milliseconds for analysis
        data_ms = self.convert_to_milliseconds(data)

        # Calculate statistics for each stage
        timing_columns = ["Meshing", "Tracing", "Postprocessing"]
        stats = {}

        for col in timing_columns:
            if col in data_ms.columns:
                stats[col] = {
                    "mean": data_ms[col].mean(),
                    "std": data_ms[col].std(),
                    "min": data_ms[col].min(),
                    "max": data_ms[col].max(),
                    "median": data_ms[col].median(),
                }
                print(f"\n{col}:")
                print(f"  Mean: {stats[col]['mean']:.2f} ms")
                print(f"  Std:  {stats[col]['std']:.2f} ms")
                print(f"  Min:  {stats[col]['min']:.2f} ms")
                print(f"  Max:  {stats[col]['max']:.2f} ms")
                print(f"  Median: {stats[col]['median']:.2f} ms")

        # Calculate total time per run
        if all(col in data_ms.columns for col in timing_columns):
            data_ms["Total"] = data_ms[timing_columns].sum(axis=1)
            total_stats = {
                "mean": data_ms["Total"].mean(),
                "std": data_ms["Total"].std(),
                "min": data_ms["Total"].min(),
                "max": data_ms["Total"].max(),
                "median": data_ms["Total"].median(),
            }
            print(f"\nTotal Time:")
            print(f"  Mean: {total_stats['mean']:.2f} ms")
            print(f"  Std:  {total_stats['std']:.2f} ms")
            print(f"  Min:  {total_stats['min']:.2f} ms")
            print(f"  Max:  {total_stats['max']:.2f} ms")
            print(f"  Median: {total_stats['median']:.2f} ms")

        return data_ms, stats

    def create_comparison_bar_plot(self, gpu_data_ms, cpu_data_ms):
        """Create bar plots comparing GPU vs CPU performance for each stage."""
        timing_columns = ["Meshing", "Tracing", "Postprocessing"]

        # Calculate means for each stage
        gpu_means = {}
        cpu_means = {}
        gpu_stds = {}
        cpu_stds = {}

        for col in timing_columns:
            if col in gpu_data_ms.columns:
                gpu_means[col] = gpu_data_ms[col].mean()
                gpu_stds[col] = gpu_data_ms[col].std()
            if col in cpu_data_ms.columns:
                cpu_means[col] = cpu_data_ms[col].mean()
                cpu_stds[col] = cpu_data_ms[col].std()

        # Create the bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "GPU vs CPU Single Benchmark Performance Comparison",
            fontsize=16,
            fontweight="bold",
        )

        # Individual stage comparisons
        for i, stage in enumerate(timing_columns):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            if stage in gpu_means and stage in cpu_means:
                categories = ["GPU", "CPU"]
                means = [gpu_means[stage], cpu_means[stage]]
                stds = [gpu_stds[stage], cpu_stds[stage]]
                colors = ["#2E86AB", "#A23B72"]

                bars = ax.bar(
                    categories, means, yerr=stds, capsize=5, color=colors, alpha=0.8
                )

                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std,
                        f"{mean:.1f} ms",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                ax.set_ylabel("Time (ms)")
                ax.set_title(f"{stage} Performance", fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")

                # Calculate and display speedup
                if gpu_means[stage] > 0:
                    speedup = cpu_means[stage] / gpu_means[stage]
                    ax.text(
                        0.5,
                        max(means) * 0.8,
                        f"CPU/GPU Ratio: {speedup:.2f}x",
                        ha="center",
                        va="center",
                        transform=ax.transData,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                        fontweight="bold",
                    )

        # Total time comparison
        ax = axes[1, 1]
        if all(stage in gpu_means and stage in cpu_means for stage in timing_columns):
            gpu_total = sum(gpu_means[stage] for stage in timing_columns)
            cpu_total = sum(cpu_means[stage] for stage in timing_columns)
            gpu_total_std = np.sqrt(
                sum(gpu_stds[stage] ** 2 for stage in timing_columns)
            )
            cpu_total_std = np.sqrt(
                sum(cpu_stds[stage] ** 2 for stage in timing_columns)
            )

            categories = ["GPU", "CPU"]
            totals = [gpu_total, cpu_total]
            total_stds = [gpu_total_std, cpu_total_std]
            colors = ["#2E86AB", "#A23B72"]

            bars = ax.bar(
                categories, totals, yerr=total_stds, capsize=5, color=colors, alpha=0.8
            )

            # Add value labels on bars
            for bar, total, std in zip(bars, totals, total_stds):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std,
                    f"{total:.1f} ms",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_ylabel("Time (ms)")
            ax.set_title("Total Processing Time", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            # Calculate and display overall speedup
            if gpu_total > 0:
                overall_speedup = cpu_total / gpu_total
                ax.text(
                    0.5,
                    max(totals) * 0.8,
                    f"Overall CPU/GPU Ratio: {overall_speedup:.2f}x",
                    ha="center",
                    va="center",
                    transform=ax.transData,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                    fontweight="bold",
                )

        plt.tight_layout()
        return fig

    def create_performance_breakdown_plot(self, gpu_data_ms, cpu_data_ms):
        """Create pie charts showing performance breakdown for GPU and CPU."""
        timing_columns = ["Meshing", "Tracing", "Postprocessing"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Performance Breakdown: GPU vs CPU", fontsize=16, fontweight="bold"
        )

        # GPU breakdown
        ax = axes[0]
        gpu_means = [
            gpu_data_ms[col].mean()
            for col in timing_columns
            if col in gpu_data_ms.columns
        ]
        gpu_labels = [col for col in timing_columns if col in gpu_data_ms.columns]
        colors = plt.cm.Set3(np.linspace(0, 1, len(gpu_labels)))

        wedges, texts, autotexts = ax.pie(
            gpu_means,
            labels=gpu_labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title("GPU Performance Breakdown", fontweight="bold")

        # CPU breakdown
        ax = axes[1]
        cpu_means = [
            cpu_data_ms[col].mean()
            for col in timing_columns
            if col in cpu_data_ms.columns
        ]
        cpu_labels = [col for col in timing_columns if col in cpu_data_ms.columns]

        wedges, texts, autotexts = ax.pie(
            cpu_means,
            labels=cpu_labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax.set_title("CPU Performance Breakdown", fontweight="bold")

        plt.tight_layout()
        return fig

    def create_speedup_analysis_plot(self, gpu_data_ms, cpu_data_ms):
        """Create speedup analysis visualization."""
        timing_columns = ["Meshing", "Tracing", "Postprocessing"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Speedup Analysis: CPU vs GPU", fontsize=16, fontweight="bold")

        # Calculate speedup factors
        speedups = {}
        for stage in timing_columns:
            if stage in gpu_data_ms.columns and stage in cpu_data_ms.columns:
                gpu_mean = gpu_data_ms[stage].mean()
                cpu_mean = cpu_data_ms[stage].mean()
                if gpu_mean > 0:
                    speedups[stage] = cpu_mean / gpu_mean

        # Speedup bar chart
        ax = axes[0]
        stages = list(speedups.keys())
        speedup_values = list(speedups.values())
        colors = ["green" if x > 1 else "red" for x in speedup_values]

        bars = ax.bar(stages, speedup_values, color=colors, alpha=0.7)
        ax.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="No Speedup")
        ax.set_ylabel("Speedup Factor (CPU time / GPU time)")
        ax.set_title("Stage-wise Speedup Analysis", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        # Add value labels on bars
        for bar, value in zip(bars, speedup_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}x",
                ha="center",
                va="bottom" if value > 0 else "top",
                fontweight="bold",
            )

        # Performance comparison in log scale
        ax = axes[1]
        x_pos = np.arange(len(stages))
        width = 0.35

        gpu_means = [gpu_data_ms[stage].mean() for stage in stages]
        cpu_means = [cpu_data_ms[stage].mean() for stage in stages]

        ax.bar(
            x_pos - width / 2, gpu_means, width, label="GPU", color="#2E86AB", alpha=0.8
        )
        ax.bar(
            x_pos + width / 2, cpu_means, width, label="CPU", color="#A23B72", alpha=0.8
        )

        ax.set_xlabel("Processing Stage")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Performance Comparison (Log Scale)", fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_summary_report(self, gpu_data_ms, cpu_data_ms):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("VIENNAPS SINGLE BENCHMARK ANALYSIS SUMMARY")
        print("=" * 80)

        timing_columns = ["Meshing", "Tracing", "Postprocessing"]

        # Calculate means and speedups
        print(f"\nPERFORMANCE COMPARISON:")
        print(
            f"{'Stage':<15} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10} {'Winner'}"
        )
        print("-" * 60)

        total_gpu = 0
        total_cpu = 0

        for stage in timing_columns:
            if stage in gpu_data_ms.columns and stage in cpu_data_ms.columns:
                gpu_mean = gpu_data_ms[stage].mean()
                cpu_mean = cpu_data_ms[stage].mean()
                speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0
                winner = "GPU" if speedup > 1 else "CPU"

                print(
                    f"{stage:<15} {gpu_mean:<12.2f} {cpu_mean:<12.2f} {speedup:<10.2f} {winner}"
                )

                total_gpu += gpu_mean
                total_cpu += cpu_mean

        overall_speedup = total_cpu / total_gpu if total_gpu > 0 else 0
        overall_winner = "GPU" if overall_speedup > 1 else "CPU"

        print("-" * 60)
        print(
            f"{'TOTAL':<15} {total_gpu:<12.2f} {total_cpu:<12.2f} {overall_speedup:<10.2f} {overall_winner}"
        )

        # Performance variability analysis
        print(f"\nPERFORMANCE VARIABILITY:")
        for stage in timing_columns:
            if stage in gpu_data_ms.columns:
                gpu_cv = (gpu_data_ms[stage].std() / gpu_data_ms[stage].mean()) * 100
                print(f"GPU {stage} coefficient of variation: {gpu_cv:.2f}%")
            if stage in cpu_data_ms.columns:
                cpu_cv = (cpu_data_ms[stage].std() / cpu_data_ms[stage].mean()) * 100
                print(f"CPU {stage} coefficient of variation: {cpu_cv:.2f}%")

        # Ray count analysis (if available)
        if "NumberOfTraces" in cpu_data_ms.columns:
            avg_rays = cpu_data_ms["NumberOfTraces"].mean()
            print(f"\nRAY TRACING ANALYSIS:")
            print(f"Average number of traces (CPU): {avg_rays:,.0f}")

    def save_processed_data(
        self, gpu_data_ms, cpu_data_ms, output_dir="single_benchmark_analysis"
    ):
        """Save processed data and summary statistics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save processed data
        if gpu_data_ms is not None:
            gpu_data_ms.to_csv(
                output_path / "gpu_single_benchmark_processed.csv", index=False
            )
        if cpu_data_ms is not None:
            cpu_data_ms.to_csv(
                output_path / "cpu_single_benchmark_processed.csv", index=False
            )

        print(f"\nProcessed data saved to: {output_path}")

    def run_analysis(self, save_plots=True, save_data=True):
        """Run the complete single benchmark analysis."""
        print("Starting ViennaPS Single Benchmark Analysis...")

        # Load data
        self.load_data()

        if self.gpu_data is None and self.cpu_data is None:
            print("Error: Could not load any benchmark data. Please check file paths.")
            return

        # Calculate statistics
        gpu_data_ms, gpu_stats = None, None
        cpu_data_ms, cpu_stats = None, None

        if self.gpu_data is not None:
            gpu_data_ms, gpu_stats = self.calculate_statistics(self.gpu_data, "GPU")
        if self.cpu_data is not None:
            cpu_data_ms, cpu_stats = self.calculate_statistics(self.cpu_data, "CPU")

        # Generate plots (only if both datasets are available)
        if save_plots and gpu_data_ms is not None and cpu_data_ms is not None:
            plt.style.use(
                "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
            )

            # Comparison bar plot
            fig1 = self.create_comparison_bar_plot(gpu_data_ms, cpu_data_ms)
            fig1.savefig(
                "single_benchmark_comparison.png", dpi=300, bbox_inches="tight"
            )

            # Performance breakdown plot
            fig2 = self.create_performance_breakdown_plot(gpu_data_ms, cpu_data_ms)
            fig2.savefig("single_benchmark_breakdown.png", dpi=300, bbox_inches="tight")

            # Speedup analysis plot
            fig3 = self.create_speedup_analysis_plot(gpu_data_ms, cpu_data_ms)
            fig3.savefig("single_benchmark_speedup.png", dpi=300, bbox_inches="tight")

            print("\nPlots saved:")
            print("- single_benchmark_comparison.png")
            print("- single_benchmark_breakdown.png")
            print("- single_benchmark_speedup.png")

            plt.show()

        # Generate summary report (only if both datasets are available)
        if gpu_data_ms is not None and cpu_data_ms is not None:
            self.generate_summary_report(gpu_data_ms, cpu_data_ms)

        # Save processed data
        if save_data:
            self.save_processed_data(gpu_data_ms, cpu_data_ms)


def main():
    """Main function to run the single benchmark analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze ViennaPS single benchmark results"
    )
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

    analyzer = SingleBenchmarkAnalyzer(args.benchmark_dir)
    analyzer.run_analysis(save_plots=not args.no_plots, save_data=not args.no_save)


if __name__ == "__main__":
    main()
