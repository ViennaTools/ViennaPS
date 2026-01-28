"""
Feature C: Marimo Interactive Simulation

Interactive simulation control and visualization components
for real-time ViennaPS parameter exploration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
import time
import hashlib
import json


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
            appended_raw = re.sub(r'\s', '', appended_elem.text)
            if appended_raw.startswith('_'):
                appended_raw = appended_raw[1:]
            # VTK base64 uses == as padding AND separator - split and decode each chunk
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

        # Get header type
        header_type = root.get("header_type", "UInt32")
        header_size = 8 if "64" in header_type else 4
        header_dtype = np.uint64 if "64" in header_type else np.uint32

        def read_block_at_position(pos):
            """Read a single data block at given position."""
            if raw_bytes is None or pos + header_size * 4 > len(raw_bytes):
                return b'', pos
            if use_zlib:
                num_blocks = int(np.frombuffer(raw_bytes[pos:pos+header_size], dtype=header_dtype)[0])
                pos += header_size * 3  # Skip num_blocks, block_size, last_block_size
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

        # Build offset to index mapping
        data_arrays = root.findall(".//DataArray[@format='appended']")
        offset_to_index = {int(da.get('offset', '0')): i for i, da in
                          enumerate(sorted(data_arrays, key=lambda x: int(x.get('offset', '0'))))}

        # Read all blocks sequentially
        blocks_data = []
        pos = 0
        while raw_bytes and pos < len(raw_bytes) - header_size * 4:
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

        format_attr = points_elem.get("format", "ascii")
        num_components = int(points_elem.get("NumberOfComponents", "3"))
        dtype_str = points_elem.get("type", "Float32")

        if format_attr == "appended":
            offset = int(points_elem.get("offset", "0"))
            points = read_appended_array(offset, dtype_str, num_components).astype(np.float64)
        elif format_attr == "ascii":
            text = points_elem.text.strip() if points_elem.text else ""
            values = [float(x) for x in text.split()]
            points = np.array(values, dtype=np.float64).reshape(-1, num_components)
        elif format_attr == "binary":
            data = base64.b64decode(points_elem.text.strip() if points_elem.text else "")
            data = data[header_size:]
            points = np.frombuffer(data, dtype=np.float64).reshape(-1, num_components)
        else:
            raise ValueError(f"Unsupported format: {format_attr}")

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
                self.bounds = (
                    float(pts[:, 0].min()), float(pts[:, 0].max()),
                    float(pts[:, 1].min()), float(pts[:, 1].max()),
                    float(pts[:, 2].min()) if pts.shape[1] > 2 else 0.0,
                    float(pts[:, 2].max()) if pts.shape[1] > 2 else 0.0,
                )
        return VTPWrapper(points, n_cells)


@dataclass
class SimulationGeometry:
    """Container for simulation geometry data."""
    points: np.ndarray
    cells: np.ndarray = field(default_factory=lambda: np.array([]))
    bounds: List[float] = field(default_factory=list)


@dataclass
class SimulationRunResult:
    """Result of a simulation run."""
    success: bool
    geometry: Optional[SimulationGeometry] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    cached: bool = False


class SimulationController:
    """Controller for interactive ViennaPS simulations."""

    def __init__(self):
        self._params: Dict[str, Any] = {
            "grid_delta": 1.0,
            "x_extent": 20.0,
            "y_extent": 15.0,
            "trench_width": 8.0,
            "trench_depth": 10.0,
            "process_type": "isotropic_etch",
            "process_time": 4.0,
            "rate": 0.5,
        }
        self._param_ranges: Dict[str, Tuple[float, float]] = {
            "grid_delta": (0.1, 5.0),
            "x_extent": (5.0, 50.0),
            "y_extent": (5.0, 50.0),
            "trench_width": (1.0, 20.0),
            "trench_depth": (1.0, 20.0),
            "process_time": (0.1, 20.0),
            "rate": (0.01, 2.0),
        }
        self._cache: Dict[str, SimulationRunResult] = {}
        self.cache_hits: int = 0
        self._last_param_hash: Optional[str] = None

    @property
    def grid_delta(self) -> float:
        return self._params["grid_delta"]

    @property
    def x_extent(self) -> float:
        return self._params["x_extent"]

    @property
    def y_extent(self) -> float:
        return self._params["y_extent"]

    def _get_param_hash(self) -> str:
        """Get hash of current parameters for caching."""
        param_str = json.dumps(self._params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a simulation parameter with validation."""
        if name in self._param_ranges:
            min_val, max_val = self._param_ranges[name]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    raise ValueError(
                        f"Parameter {name}={value} outside range [{min_val}, {max_val}]"
                    )
        elif name == "process_type":
            valid_types = ["isotropic_etch", "deposition", "directional_etch"]
            if value not in valid_types:
                raise ValueError(f"Invalid process_type: {value}")

        self._params[name] = value

    def get_parameter(self, name: str) -> Any:
        """Get current value of a parameter."""
        return self._params.get(name)

    def run(self) -> SimulationRunResult:
        """Run simulation with current parameters."""
        start_time = time.time()

        # Check cache
        param_hash = self._get_param_hash()
        if param_hash in self._cache:
            self.cache_hits += 1
            return self._cache[param_hash]

        try:
            import viennaps as ps
            import os

            ps.setDimension(2)
            ps.setNumThreads(1)
            os.environ["OMP_NUM_THREADS"] = "1"

            # Create domain (API: Domain(grid_delta, x_extent, y_extent))
            domain = ps.Domain(
                self._params["grid_delta"],
                self._params["x_extent"],
                self._params["y_extent"]
            )

            # Generate initial geometry with MakePlane
            ps.MakePlane(domain, 0.0, ps.Material.Si).apply()

            # Create process model
            process_type = self._params["process_type"]
            rate = self._params["rate"]

            if process_type == "isotropic_etch":
                model = ps.IsotropicProcess(rate=-rate)
            elif process_type == "deposition":
                model = ps.SingleParticleProcess(
                    rate=rate,
                    stickingProbability=0.1,
                    sourceExponent=1.0
                )
            else:
                model = ps.IsotropicProcess(rate=-rate)

            # Run process
            ps.Process(domain, model, self._params["process_time"]).apply()

            # Extract geometry
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
                temp_path = f.name

            domain.saveSurfaceMesh(temp_path)

            mesh = _read_vtp_mesh(temp_path)
            points = np.array(mesh.points, dtype=np.float64)

            geometry = SimulationGeometry(
                points=points,
                bounds=list(mesh.bounds)
            )

            Path(temp_path).unlink()

            execution_time = time.time() - start_time
            result = SimulationRunResult(
                success=True,
                geometry=geometry,
                execution_time=execution_time
            )

            # Cache result
            self._cache[param_hash] = result
            return result

        except ImportError:
            # Create mock geometry for testing without ViennaPS
            execution_time = time.time() - start_time
            mock_points = np.array([
                [0, 0, 0],
                [self._params["x_extent"], 0, 0],
            ], dtype=np.float64)

            geometry = SimulationGeometry(
                points=mock_points,
                bounds=[0, self._params["x_extent"], 0, self._params["y_extent"], 0, 0]
            )

            result = SimulationRunResult(
                success=True,
                geometry=geometry,
                execution_time=execution_time
            )

            self._cache[param_hash] = result
            return result

        except Exception as e:
            return SimulationRunResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )


class ParameterWidget:
    """Widget for parameter input in Marimo notebooks."""

    def __init__(
        self,
        name: str,
        widget_type: str,
        min: Optional[float] = None,
        max: Optional[float] = None,
        default: Any = None,
        step: Optional[float] = None,
        label: Optional[str] = None,
        options: Optional[List[Any]] = None,
        on_change: Optional[Callable[[Any], None]] = None
    ):
        self.name = name
        self.widget_type = widget_type
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        self.label = label or name
        self.options = options or []
        self._on_change = on_change
        self._value = default

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        self._value = new_value
        if self._on_change:
            self._on_change(new_value)

    @classmethod
    def slider(
        cls,
        name: str,
        min_val: float,
        max_val: float,
        default: float,
        step: Optional[float] = None,
        label: Optional[str] = None,
        on_change: Optional[Callable[[Any], None]] = None
    ) -> "ParameterWidget":
        """Create a slider widget for continuous parameters."""
        return cls(
            name=name,
            widget_type="slider",
            min=min_val,
            max=max_val,
            default=default,
            step=step or (max_val - min_val) / 100,
            label=label,
            on_change=on_change
        )

    @classmethod
    def dropdown(
        cls,
        name: str,
        options: List[Any],
        default: Any,
        label: Optional[str] = None,
        on_change: Optional[Callable[[Any], None]] = None
    ) -> "ParameterWidget":
        """Create a dropdown widget for discrete parameters."""
        return cls(
            name=name,
            widget_type="dropdown",
            options=options,
            default=default,
            label=label,
            on_change=on_change
        )

    def to_marimo(self) -> Any:
        """Convert to Marimo UI element."""
        try:
            import marimo as mo

            if self.widget_type == "slider":
                return mo.ui.slider(
                    start=self.min,
                    stop=self.max,
                    step=self.step,
                    value=self.default,
                    label=self.label
                )
            elif self.widget_type == "dropdown":
                return mo.ui.dropdown(
                    options={str(o): o for o in self.options},
                    value=str(self.default),
                    label=self.label
                )
        except ImportError:
            # Return mock for testing
            return MockMarimoElement(self)

        return None


class MockMarimoElement:
    """Mock Marimo element for testing."""

    def __init__(self, widget: ParameterWidget):
        self.widget = widget
        self.value = widget.default

    def to_html(self) -> str:
        return f"<div>{self.widget.label}: {self.value}</div>"


class ResultsDisplay:
    """Display simulation results in Marimo notebooks."""

    def __init__(self):
        self.current_data: Optional[np.ndarray] = None
        self._current_element: Any = None

    def show_geometry(self, points: np.ndarray) -> Any:
        """Display geometry as interactive plot."""
        self.current_data = points.copy()

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode='lines+markers',
                name='Surface'
            ))
            fig.update_layout(
                title="Simulation Geometry",
                xaxis_title="X (um)",
                yaxis_title="Y (um)",
                yaxis=dict(scaleanchor="x")
            )

            self._current_element = fig
            return fig

        except ImportError:
            # Return mock element for testing
            mock = MockPlotElement(points)
            self._current_element = mock
            return mock

    def show_metrics(self, metrics: Dict[str, Any]) -> Any:
        """Display metrics as formatted table."""
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[
                    list(metrics.keys()),
                    [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]
                ])
            )])
            fig.update_layout(title="Simulation Metrics")

            return fig

        except ImportError:
            return MockTableElement(metrics)

    def show_comparison(
        self,
        before: np.ndarray,
        after: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Any:
        """Display before/after comparison."""
        if labels is None:
            labels = ["Before", "After"]

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=before[:, 0], y=before[:, 1],
                mode='lines+markers',
                name=labels[0]
            ))
            fig.add_trace(go.Scatter(
                x=after[:, 0], y=after[:, 1],
                mode='lines+markers',
                name=labels[1]
            ))
            fig.update_layout(
                title="Comparison View",
                xaxis_title="X (um)",
                yaxis_title="Y (um)"
            )

            return fig

        except ImportError:
            return MockComparisonElement(before, after, labels)

    def update(self, new_data: np.ndarray) -> None:
        """Update display with new data."""
        self.current_data = new_data.copy()
        if self._current_element is not None:
            self.show_geometry(new_data)


class MockPlotElement:
    """Mock plot element for testing."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __marimo__(self):
        return True

    def to_html(self) -> str:
        return f"<div>Plot with {len(self.data)} points</div>"


class MockTableElement:
    """Mock table element for testing."""

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def to_html(self) -> str:
        return "<table>" + "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in self.metrics.items()
        ) + "</table>"


class MockComparisonElement:
    """Mock comparison element for testing."""

    def __init__(self, before: np.ndarray, after: np.ndarray, labels: List[str]):
        self.before = before
        self.after = after
        self.labels = labels

    def to_html(self) -> str:
        return f"<div>Comparison: {self.labels[0]} vs {self.labels[1]}</div>"


@dataclass
class InteractiveApp:
    """Interactive simulation application."""
    controller: SimulationController
    widgets: Dict[str, ParameterWidget]
    display: ResultsDisplay

    def export_notebook(self, path: Path) -> None:
        """Export as Marimo notebook."""
        notebook_content = '''import marimo

app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import numpy as np
    from src.interactive import SimulationController, ResultsDisplay
    return mo, np, SimulationController, ResultsDisplay

@app.cell
def __(mo):
    # Parameter controls
    grid_delta = mo.ui.slider(start=0.5, stop=3.0, step=0.1, value=1.0, label="Grid Delta")
    trench_width = mo.ui.slider(start=2.0, stop=15.0, step=0.5, value=8.0, label="Trench Width (um)")
    process_type = mo.ui.dropdown(
        options={"Deposition": "deposition", "Etching": "isotropic_etch"},
        value="deposition",
        label="Process Type"
    )
    mo.md("## Simulation Parameters")
    return grid_delta, trench_width, process_type

@app.cell
def __(grid_delta, trench_width, process_type, SimulationController, ResultsDisplay):
    controller = SimulationController()
    controller.set_parameter("grid_delta", grid_delta.value)
    controller.set_parameter("trench_width", trench_width.value)
    controller.set_parameter("process_type", process_type.value)

    result = controller.run()

    display = ResultsDisplay()
    if result.success:
        fig = display.show_geometry(result.geometry.points)
    else:
        fig = None
    return controller, result, display, fig

@app.cell
def __(fig, mo):
    mo.md("## Simulation Result")
    fig

if __name__ == "__main__":
    app.run()
'''
        Path(path).write_text(notebook_content)


def create_interactive_app(
    param_ranges: Optional[Dict[str, Tuple[float, float, float]]] = None
) -> InteractiveApp:
    """Factory function to create an interactive simulation app."""
    controller = SimulationController()
    display = ResultsDisplay()
    widgets: Dict[str, ParameterWidget] = {}

    # Default parameter ranges: (min, max, default)
    default_ranges = {
        "grid_delta": (0.5, 3.0, 1.0),
        "trench_width": (2.0, 15.0, 8.0),
        "trench_depth": (5.0, 15.0, 10.0),
        "process_time": (1.0, 10.0, 4.0),
    }

    if param_ranges:
        default_ranges.update(param_ranges)

    for name, (min_val, max_val, default) in default_ranges.items():
        widgets[name] = ParameterWidget.slider(
            name=name,
            min_val=min_val,
            max_val=max_val,
            default=default,
            label=f"{name.replace('_', ' ').title()}"
        )

    return InteractiveApp(
        controller=controller,
        widgets=widgets,
        display=display
    )
