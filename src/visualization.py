"""
Feature A: ViennaPS Visualization Pipeline

Comprehensive visualization tools for ViennaPS simulation outputs.
Converts VTP files to interactive Plotly figures with comparison views.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class Geometry:
    """Container for geometry data from VTP files."""
    points: np.ndarray
    cells: np.ndarray
    bounds: List[float]  # [xmin, xmax, ymin, ymax, zmin, zmax]

    @property
    def num_points(self) -> int:
        return len(self.points)


class VTPReader:
    """Reader for VTK PolyData (VTP) files."""

    def read(self, path: Path) -> Geometry:
        """Read VTP file and return Geometry object."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"VTP file not found: {path}")

        try:
            import pyvista as pv
            mesh = pv.read(str(path))
            points = np.array(mesh.points, dtype=np.float64)
            cells = np.array(mesh.cells) if hasattr(mesh, 'cells') else np.array([])
            bounds = list(mesh.bounds)  # (xmin, xmax, ymin, ymax, zmin, zmax)
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
                # VTK base64 uses == as padding AND separator
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

            cells = np.array([])
            bounds = [
                float(np.min(points[:, 0])), float(np.max(points[:, 0])),
                float(np.min(points[:, 1])), float(np.max(points[:, 1])),
                float(np.min(points[:, 2])) if points.shape[1] > 2 else 0.0,
                float(np.max(points[:, 2])) if points.shape[1] > 2 else 0.0,
            ]

        return Geometry(points=points, cells=cells, bounds=bounds)


class SimulationVisualizer:
    """Create interactive Plotly visualizations of simulation results."""

    def __init__(self):
        self.default_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'grid': '#e0e0e0'
        }

    def plot_profile(
        self,
        points: np.ndarray,
        units: str = "um",
        title: str = "Surface Profile"
    ) -> go.Figure:
        """Create 2D cross-section profile plot."""
        # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='Surface',
            line=dict(color=self.default_colors['primary'], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=title,
            xaxis_title=f"X ({units})",
            yaxis_title=f"Y ({units})",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor=self.default_colors['grid']),
            yaxis=dict(showgrid=True, gridcolor=self.default_colors['grid'], scaleanchor="x"),
        )

        return fig

    def plot_surface(
        self,
        points: np.ndarray,
        units: str = "um",
        title: str = "3D Surface View"
    ) -> go.Figure:
        """Create 3D surface mesh visualization."""
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2] if points.shape[1] > 2 else np.zeros(len(x))

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+lines',
            name='Surface',
            marker=dict(size=3, color=self.default_colors['primary']),
            line=dict(color=self.default_colors['primary'], width=2)
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"X ({units})",
                yaxis_title=f"Y ({units})",
                zaxis_title=f"Z ({units})",
                aspectmode='data'
            ),
            showlegend=True,
        )

        return fig


class ComparisonView:
    """Create comparison visualizations between simulation states."""

    def __init__(self):
        self.colors = {
            'before': '#1f77b4',  # blue
            'after': '#ff7f0e',   # orange
        }

    def side_by_side(
        self,
        points_before: np.ndarray,
        points_after: np.ndarray,
        titles: Optional[List[str]] = None
    ) -> go.Figure:
        """Create side-by-side comparison of two geometries."""
        if titles is None:
            titles = ["Before", "After"]

        fig = make_subplots(rows=1, cols=2, subplot_titles=titles)

        # Before plot
        fig.add_trace(
            go.Scatter(
                x=points_before[:, 0],
                y=points_before[:, 1],
                mode='lines+markers',
                name=titles[0],
                line=dict(color=self.colors['before'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

        # After plot
        fig.add_trace(
            go.Scatter(
                x=points_after[:, 0],
                y=points_after[:, 1],
                mode='lines+markers',
                name=titles[1],
                line=dict(color=self.colors['after'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="Side-by-Side Comparison",
            showlegend=True,
        )

        # Match y-axis scales
        y_min = min(np.min(points_before[:, 1]), np.min(points_after[:, 1]))
        y_max = max(np.max(points_before[:, 1]), np.max(points_after[:, 1]))
        margin = (y_max - y_min) * 0.1
        fig.update_yaxes(range=[y_min - margin, y_max + margin])

        return fig

    def overlay(
        self,
        points_before: np.ndarray,
        points_after: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> go.Figure:
        """Create overlay comparison with different colors."""
        if labels is None:
            labels = ["Before", "After"]

        fig = go.Figure()

        # Add before trace
        fig.add_trace(go.Scatter(
            x=points_before[:, 0],
            y=points_before[:, 1],
            mode='lines+markers',
            name=labels[0],
            line=dict(color=self.colors['before'], width=2),
            marker=dict(size=4)
        ))

        # Add after trace
        fig.add_trace(go.Scatter(
            x=points_after[:, 0],
            y=points_after[:, 1],
            mode='lines+markers',
            name=labels[1],
            line=dict(color=self.colors['after'], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title="Overlay Comparison",
            xaxis_title="X (um)",
            yaxis_title="Y (um)",
            showlegend=True,
            hovermode='closest',
            yaxis=dict(scaleanchor="x"),
        )

        return fig

    def compute_difference(
        self,
        points_before: np.ndarray,
        points_after: np.ndarray
    ) -> Dict[str, float]:
        """Compute geometry differences between two states."""
        # Simple approach: compare y-coordinates at matching x positions
        # For points with same x, compute y difference
        if len(points_before) != len(points_after):
            # Interpolate to same length for comparison
            from scipy import interpolate
            x_before = points_before[:, 0]
            y_before = points_before[:, 1]
            x_after = points_after[:, 0]
            y_after = points_after[:, 1]

            # Common x range
            x_common = np.linspace(
                max(x_before.min(), x_after.min()),
                min(x_before.max(), x_after.max()),
                100
            )

            f_before = interpolate.interp1d(x_before, y_before, kind='linear', fill_value='extrapolate')
            f_after = interpolate.interp1d(x_after, y_after, kind='linear', fill_value='extrapolate')

            y_before_interp = f_before(x_common)
            y_after_interp = f_after(x_common)

            diff = np.abs(y_after_interp - y_before_interp)
        else:
            # Direct comparison
            diff = np.abs(points_after[:, 1] - points_before[:, 1])

        return {
            'max_change': float(np.max(diff)),
            'mean_change': float(np.mean(diff)),
            'min_change': float(np.min(diff)),
            'std_change': float(np.std(diff))
        }


def export_to_html(fig: go.Figure, path: Path, include_plotlyjs: bool = True) -> None:
    """Export Plotly figure to HTML file."""
    path = Path(path)
    fig.write_html(
        str(path),
        include_plotlyjs='cdn' if include_plotlyjs else False,
        full_html=True
    )


def visualize_simulation_output(
    output_dir: Path,
    pattern: str = "*.vtp"
) -> Dict[str, go.Figure]:
    """Create visualizations for all VTP files in a directory."""
    reader = VTPReader()
    visualizer = SimulationVisualizer()

    output_dir = Path(output_dir)
    figures = {}

    for vtp_file in sorted(output_dir.glob(pattern)):
        try:
            geometry = reader.read(vtp_file)
            fig = visualizer.plot_profile(
                geometry.points,
                title=vtp_file.stem.replace('_', ' ').title()
            )
            figures[vtp_file.stem] = fig
        except Exception as e:
            print(f"Warning: Could not visualize {vtp_file}: {e}")

    return figures


def create_process_comparison(
    output_dir: Path,
    initial_pattern: str = "*_initial.vtp",
    final_pattern: str = "*_deposited.vtp"
) -> Optional[go.Figure]:
    """Create comparison between initial and final states."""
    reader = VTPReader()
    comparison = ComparisonView()

    output_dir = Path(output_dir)

    initial_files = list(output_dir.glob(initial_pattern))
    if not initial_files:
        return None

    initial_file = initial_files[0]
    prefix = initial_file.stem.replace('_initial', '')

    # Find corresponding final file
    final_files = [f for f in output_dir.glob("*.vtp")
                   if prefix in f.stem and 'initial' not in f.stem]

    if not final_files:
        return None

    final_file = final_files[0]

    initial_geom = reader.read(initial_file)
    final_geom = reader.read(final_file)

    return comparison.overlay(
        initial_geom.points,
        final_geom.points,
        labels=["Initial", final_file.stem.replace(prefix + '_', '').title()]
    )
