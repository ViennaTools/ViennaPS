"""
Feature B: Parameter Sweep Framework

Framework for running ViennaPS simulations across parameter spaces
with metrics collection and parallel execution support.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Callable, Union
import numpy as np
import itertools
import csv
import tempfile
import os


def _read_vtp_mesh(path):
    """Read VTP mesh with pyvista, falling back to XML parser if VTK unavailable."""
    try:
        import pyvista as pv
        return pv.read(str(path))
    except ImportError:
        # Fallback to VTP XML parser supporting appended/binary/ascii formats
        import xml.etree.ElementTree as ET
        import base64
        import zlib
        import re

        tree = ET.parse(str(path))
        root = tree.getroot()

        # Check for appended data (common VTK format)
        appended_elem = root.find(".//AppendedData")
        raw_bytes = None
        if appended_elem is not None and appended_elem.text:
            # Strip leading underscore marker and whitespace
            appended_raw = re.sub(r'\s', '', appended_elem.text)
            if appended_raw.startswith('_'):
                appended_raw = appended_raw[1:]

            # VTK base64 format uses == as both padding AND separator
            # Split by == and decode each chunk, then concatenate
            chunks = appended_raw.split('==')
            raw_bytes = b''
            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                if i < len(chunks) - 1:
                    chunk_padded = chunk + '=='
                else:
                    padding = 4 - len(chunk) % 4
                    chunk_padded = chunk + '=' * padding if padding < 4 else chunk
                try:
                    raw_bytes += base64.b64decode(chunk_padded)
                except Exception:
                    pass

        # Check if zlib compression is used
        compressor = root.get("compressor", "")
        use_zlib = "zlib" in compressor.lower()

        # Get header type (default UInt32 = 4 bytes)
        header_type = root.get("header_type", "UInt32")
        header_size = 8 if "64" in header_type else 4
        header_dtype = np.uint64 if "64" in header_type else np.uint32

        def read_block_at_position(pos):
            """Read a single data block at given position, return (data_bytes, next_pos)."""
            if raw_bytes is None or pos + header_size * 4 > len(raw_bytes):
                return b'', pos

            if use_zlib:
                num_blocks = int(np.frombuffer(raw_bytes[pos:pos+header_size], dtype=header_dtype)[0])
                pos += header_size
                pos += header_size  # Skip block_size
                pos += header_size  # Skip last_block_size

                compressed_sizes = []
                for _ in range(num_blocks):
                    if pos + header_size > len(raw_bytes):
                        break
                    csize = int(np.frombuffer(raw_bytes[pos:pos+header_size], dtype=header_dtype)[0])
                    compressed_sizes.append(csize)
                    pos += header_size

                decompressed = b''
                for csize in compressed_sizes:
                    if pos + csize > len(raw_bytes):
                        break
                    try:
                        decompressed += zlib.decompress(raw_bytes[pos:pos+csize])
                    except zlib.error:
                        decompressed += raw_bytes[pos:pos+csize]
                    pos += csize
                return decompressed, pos
            else:
                data_size = int(np.frombuffer(raw_bytes[pos:pos+header_size], dtype=header_dtype)[0])
                pos += header_size
                return raw_bytes[pos:pos+data_size], pos + data_size

        # Read all blocks sequentially and index by offset order
        data_arrays = root.findall(".//DataArray[@format='appended']")
        offset_to_index = {}
        for i, da in enumerate(sorted(data_arrays, key=lambda x: int(x.get('offset', '0')))):
            offset_to_index[int(da.get('offset', '0'))] = i

        # Read all blocks in order
        blocks_data = []
        pos = 0
        while pos < len(raw_bytes) - header_size * 4:
            data, new_pos = read_block_at_position(pos)
            if new_pos == pos or len(data) == 0:
                break
            blocks_data.append(data)
            pos = new_pos

        def read_appended_array(offset, dtype_str, num_components=1):
            """Read array from appended data by offset (maps to block index)."""
            dtype_map = {
                "Float32": np.float32, "Float64": np.float64,
                "Int32": np.int32, "Int64": np.int64,
                "UInt32": np.uint32, "UInt64": np.uint64,
            }
            dtype = dtype_map.get(dtype_str, np.float32)

            block_idx = offset_to_index.get(offset, -1)
            if block_idx < 0 or block_idx >= len(blocks_data):
                return np.array([])

            data = blocks_data[block_idx]
            if len(data) == 0:
                return np.array([])
            return np.frombuffer(data, dtype=dtype).reshape(-1, num_components)

        # Find Points data
        points_elem = root.find(".//Points/DataArray")
        if points_elem is None:
            raise ValueError(f"No Points found in VTP file: {path}")

        # Parse points based on format
        format_attr = points_elem.get("format", "ascii")
        num_components = int(points_elem.get("NumberOfComponents", "3"))
        dtype_str = points_elem.get("type", "Float32")

        if format_attr == "appended":
            offset = int(points_elem.get("offset", "0"))
            points = read_appended_array(offset, dtype_str, num_components)
        elif format_attr == "ascii":
            text = points_elem.text.strip() if points_elem.text else ""
            values = [float(x) for x in text.split()]
            points = np.array(values, dtype=np.float64).reshape(-1, num_components)
        elif format_attr == "binary":
            # Base64 encoded binary inline
            data = base64.b64decode(points_elem.text.strip() if points_elem.text else "")
            data = data[header_size:]  # Skip header
            points = np.frombuffer(data, dtype=np.float64).reshape(-1, num_components)
        else:
            raise ValueError(f"Unsupported format: {format_attr}")

        # Ensure float64 for consistency
        points = points.astype(np.float64)

        # Count cells from Lines element
        lines_elem = root.find(".//Lines")
        n_cells = 0
        if lines_elem is not None:
            offsets_elem = lines_elem.find("DataArray[@Name='offsets']")
            if offsets_elem is not None:
                if offsets_elem.get("format") == "appended":
                    offset = int(offsets_elem.get("offset", "0"))
                    offsets_data = read_appended_array(offset, offsets_elem.get("type", "Int64"))
                    n_cells = len(offsets_data)
                elif offsets_elem.text:
                    n_cells = len(offsets_elem.text.strip().split())

        # Create wrapper object with same interface as pyvista mesh
        class VTPWrapper:
            def __init__(self, pts, num_cells):
                self.points = pts
                self.n_points = len(pts)
                self.n_cells = num_cells
                self.cells = []
                # Compute bounds
                self.bounds = (
                    float(pts[:, 0].min()), float(pts[:, 0].max()),
                    float(pts[:, 1].min()), float(pts[:, 1].max()),
                    float(pts[:, 2].min()) if pts.shape[1] > 2 else 0.0,
                    float(pts[:, 2].max()) if pts.shape[1] > 2 else 0.0,
                )
        return VTPWrapper(points, n_cells)


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    success: bool
    params: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    output_file: Optional[Path] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0


class ParameterSpace:
    """Define and iterate over parameter combinations."""

    def __init__(self):
        self._parameters: Dict[str, Dict[str, Any]] = {}

    def add_parameter(
        self,
        name: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        steps: Optional[int] = None,
        scale: str = "linear",
        values: Optional[List[Any]] = None
    ) -> None:
        """Add a parameter to the space.

        Args:
            name: Parameter name
            min: Minimum value (for range)
            max: Maximum value (for range)
            steps: Number of steps (for range)
            scale: 'linear' or 'log' (for range)
            values: Explicit list of values (overrides min/max/steps)
        """
        if values is not None:
            # Discrete values
            self._parameters[name] = {"type": "discrete", "values": values}
        elif min is not None and max is not None and steps is not None:
            # Range
            self._parameters[name] = {
                "type": "range",
                "min": min,
                "max": max,
                "steps": steps,
                "scale": scale
            }
        else:
            raise ValueError(f"Parameter {name}: provide either values or min/max/steps")

    def get_values(self, name: str) -> List[Any]:
        """Get the list of values for a parameter."""
        if name not in self._parameters:
            raise KeyError(f"Parameter {name} not defined")

        param = self._parameters[name]

        if param["type"] == "discrete":
            return param["values"]
        else:
            # Generate range
            if param["scale"] == "log":
                return list(np.logspace(
                    np.log10(param["min"]),
                    np.log10(param["max"]),
                    param["steps"]
                ))
            else:
                return list(np.linspace(param["min"], param["max"], param["steps"]))

    @property
    def total_count(self) -> int:
        """Total number of parameter combinations."""
        count = 1
        for name in self._parameters:
            count *= len(self.get_values(name))
        return count

    def combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all parameter combinations."""
        if not self._parameters:
            yield {}
            return

        param_names = list(self._parameters.keys())
        param_values = [self.get_values(name) for name in param_names]

        for combo in itertools.product(*param_values):
            yield dict(zip(param_names, combo))


@dataclass
class SweepConfiguration:
    """Configuration for a parameter sweep."""
    base_params: Dict[str, Any] = field(default_factory=dict)
    process_type: str = "isotropic_etch"
    output_dir: Path = field(default_factory=lambda: Path("sweep_output"))
    max_workers: int = 1
    parallel: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class SweepResults:
    """Container for sweep results with filtering and analysis."""

    def __init__(self):
        self._results: List[SimulationResult] = []

    def add(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        output_file: Optional[Path] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Add a result to the collection."""
        self._results.append(SimulationResult(
            success=success,
            params=params,
            metrics=metrics,
            output_file=output_file,
            error_message=error_message
        ))

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[SimulationResult]:
        return iter(self._results)

    def filter(self, **kwargs) -> "SweepResults":
        """Filter results by parameter values."""
        filtered = SweepResults()
        for result in self._results:
            match = all(
                result.params.get(key) == value
                for key, value in kwargs.items()
            )
            if match:
                filtered._results.append(result)
        return filtered

    def best_by(self, metric: str, maximize: bool = True) -> Optional[SimulationResult]:
        """Find the best result by a metric."""
        if not self._results:
            return None

        valid_results = [r for r in self._results if metric in r.metrics]
        if not valid_results:
            return None

        if maximize:
            return max(valid_results, key=lambda r: r.metrics[metric])
        else:
            return min(valid_results, key=lambda r: r.metrics[metric])


class SweepRunner:
    """Execute parameter sweeps."""

    def __init__(self, config: SweepConfiguration):
        self.config = config
        self._metrics_collector = MetricsCollector()

    def run_single(self, params: Dict[str, Any]) -> SimulationResult:
        """Run a single simulation with given parameters."""
        try:
            import viennaps as ps
        except ImportError:
            return SimulationResult(
                success=False,
                params=params,
                error_message="ViennaPS not available"
            )

        # Validate process type
        valid_types = ["isotropic_etch", "deposition", "directional_etch", "geometric_deposition"]
        if self.config.process_type not in valid_types:
            return SimulationResult(
                success=False,
                params=params,
                error_message=f"Invalid process type: {self.config.process_type}"
            )

        try:
            # Set 2D mode and single-threaded for macOS ARM
            ps.setDimension(2)
            ps.setNumThreads(1)
            os.environ["OMP_NUM_THREADS"] = "1"

            # Merge base params with sweep params
            all_params = {**self.config.base_params, **params}

            # Create domain (API: Domain(grid_delta, x_extent, y_extent))
            grid_delta = all_params.get("grid_delta", 1.0)
            x_extent = all_params.get("x_extent", 10.0)
            y_extent = all_params.get("y_extent", 10.0)

            domain = ps.Domain(grid_delta, x_extent, y_extent)

            # Create plane at y=0 using MakePlane
            ps.MakePlane(domain, 0.0, ps.Material.Si).apply()

            # Create process based on type
            rate = all_params.get("rate", -0.5)
            time_val = all_params.get("time", 4.0)

            if self.config.process_type == "isotropic_etch":
                model = ps.IsotropicProcess(rate=rate)
            elif self.config.process_type == "deposition":
                model = ps.SingleParticleProcess(
                    rate=-rate,  # Positive for deposition
                    stickingProbability=all_params.get("sticking_prob", 0.1),
                    sourceDirection=[0, 1, 0]
                )
            else:
                model = ps.IsotropicProcess(rate=rate)

            # Run process
            ps.Process(domain, model, time_val).apply()

            # Save output
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            param_str = "_".join(f"{k}={v}" for k, v in params.items())
            output_file = self.config.output_dir / f"sweep_{param_str}.vtp"
            domain.saveSurfaceMesh(str(output_file))

            # Extract metrics
            metrics = self._metrics_collector.extract(output_file)

            return SimulationResult(
                success=True,
                params=params,
                metrics=metrics,
                output_file=output_file
            )

        except Exception as e:
            return SimulationResult(
                success=False,
                params=params,
                error_message=str(e)
            )

    def run_sweep(
        self,
        space: ParameterSpace,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None
    ) -> SweepResults:
        """Run sweep over parameter space."""
        results = SweepResults()
        total = space.total_count

        for i, params in enumerate(space.combinations()):
            if progress_callback:
                progress_callback(i + 1, total, params)

            result = self.run_single(params)
            results.add(
                params=result.params,
                metrics=result.metrics,
                output_file=result.output_file,
                success=result.success,
                error_message=result.error_message
            )

        return results


class MetricsCollector:
    """Collect metrics from simulation outputs."""

    def extract(self, path: Path) -> Dict[str, Any]:
        """Extract metrics from VTP file."""
        try:
            mesh = _read_vtp_mesh(path)

            points = np.array(mesh.points)

            # Compute surface area (line length for 2D)
            if len(points) > 1:
                diffs = np.diff(points, axis=0)
                surface_area = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
            else:
                surface_area = 0.0

            return {
                "num_points": len(points),
                "bounds": list(mesh.bounds),
                "surface_area": surface_area,
                "y_min": float(np.min(points[:, 1])) if len(points) > 0 else 0.0,
                "y_max": float(np.max(points[:, 1])) if len(points) > 0 else 0.0,
            }
        except Exception as e:
            return {"error": str(e)}

    def compute_etch_depth(self, before_path: Path, after_path: Path) -> float:
        """Compute etch depth from before/after."""
        before = _read_vtp_mesh(before_path)
        after = _read_vtp_mesh(after_path)

        before_y_max = np.max(before.points[:, 1])
        after_y_max = np.max(after.points[:, 1])

        # Etch moves surface down, so before > after
        return abs(before_y_max - after_y_max)

    def compute_deposition_thickness(self, before_path: Path, after_path: Path) -> float:
        """Compute deposition thickness from before/after."""
        before = _read_vtp_mesh(before_path)
        after = _read_vtp_mesh(after_path)

        before_y_max = np.max(before.points[:, 1])
        after_y_max = np.max(after.points[:, 1])

        # Deposition moves surface up, so after > before
        return abs(after_y_max - before_y_max)

    def export_csv(
        self,
        results_data: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """Export results to CSV."""
        if not results_data:
            return

        # Collect all keys
        param_keys = set()
        metric_keys = set()
        for item in results_data:
            param_keys.update(item.get("params", {}).keys())
            metric_keys.update(item.get("metrics", {}).keys())

        param_keys = sorted(param_keys)
        metric_keys = sorted(metric_keys)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(param_keys + metric_keys)

            # Data rows
            for item in results_data:
                params = item.get("params", {})
                metrics = item.get("metrics", {})

                row = [params.get(k, "") for k in param_keys]
                row += [metrics.get(k, "") for k in metric_keys]
                writer.writerow(row)
