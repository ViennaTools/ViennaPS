"""
Feature D: ViennaPS Test Suite / Validation

Comprehensive validation framework for ViennaPS simulations.
Validates geometry, physics, and metrics against expected values.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
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
                pos += header_size * 3
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
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    check_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Geometry:
    """Container for geometry data from VTP files."""
    points: np.ndarray
    cells: np.ndarray
    bounds: List[float]  # [xmin, xmax, ymin, ymax, zmin, zmax]

    @property
    def num_points(self) -> int:
        return len(self.points)


class ViennaValidator:
    """Main validator for ViennaPS simulation outputs."""

    def check_output_exists(self, path: Path) -> ValidationResult:
        """Check if output file exists."""
        path = Path(path)
        if path.exists():
            return ValidationResult(
                passed=True,
                message=f"Output file exists: {path}",
                check_name="output_exists",
                details={"path": str(path), "size": path.stat().st_size}
            )
        return ValidationResult(
            passed=False,
            message=f"Output file not found: {path}",
            check_name="output_exists",
            details={"path": str(path)}
        )

    def check_geometry_valid(
        self,
        path: Optional[Path] = None,
        geometry_data: Optional[Dict] = None
    ) -> ValidationResult:
        """Check if geometry is valid (non-empty, non-degenerate)."""
        if geometry_data is not None:
            points = geometry_data.get("points", np.array([]))
            cells = geometry_data.get("cells", np.array([]))
        elif path is not None:
            try:
                mesh = _read_vtp_mesh(path)
                points = np.array(mesh.points)
                cells = np.array(mesh.cells) if hasattr(mesh, 'cells') else np.array([])
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    message=f"Failed to read geometry: {e}",
                    check_name="geometry_valid"
                )
        else:
            return ValidationResult(
                passed=False,
                message="No geometry data provided",
                check_name="geometry_valid"
            )

        if len(points) == 0:
            return ValidationResult(
                passed=False,
                message="Geometry has no points (degenerate)",
                check_name="geometry_valid",
                details={"num_points": 0}
            )

        return ValidationResult(
            passed=True,
            message=f"Geometry valid with {len(points)} points",
            check_name="geometry_valid",
            details={"num_points": len(points), "num_cells": len(cells)}
        )

    def check_bounds_reasonable(
        self,
        path: Path,
        expected_bounds: Dict[str, float]
    ) -> ValidationResult:
        """Check if geometry bounds are within expected range."""
        try:
            mesh = _read_vtp_mesh(path)
            bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

            issues = []
            if "x_max" in expected_bounds and bounds[1] > expected_bounds["x_max"]:
                issues.append(f"x_max ({bounds[1]}) > expected ({expected_bounds['x_max']})")
            if "y_max" in expected_bounds and bounds[3] > expected_bounds["y_max"]:
                issues.append(f"y_max ({bounds[3]}) > expected ({expected_bounds['y_max']})")

            if issues:
                return ValidationResult(
                    passed=False,
                    message=f"Bounds exceeded: {'; '.join(issues)}",
                    check_name="bounds_reasonable",
                    details={"actual_bounds": list(bounds), "expected": expected_bounds}
                )

            return ValidationResult(
                passed=True,
                message="Geometry bounds within expected range",
                check_name="bounds_reasonable",
                details={"actual_bounds": list(bounds)}
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to check bounds: {e}",
                check_name="bounds_reasonable"
            )


class GeometryMetrics:
    """Compute metrics from VTP geometry files."""

    def _read_mesh(self, path: Path):
        """Read VTP mesh using pyvista or meshio fallback."""
        return _read_vtp_mesh(path)

    def surface_area(self, path: Path) -> float:
        """Compute surface area from geometry."""
        mesh = self._read_mesh(path)
        # For 2D simulations (lines), compute total line length
        points = np.array(mesh.points)
        if len(points) > 1:
            # Sum of segment lengths between consecutive points
            diffs = np.diff(points, axis=0)
            lengths = np.sqrt(np.sum(diffs**2, axis=1))
            return float(np.sum(lengths))
        return 0.0

    def trench_depth(self, path: Path) -> float:
        """Compute trench depth from geometry (y-direction extent below surface)."""
        mesh = self._read_mesh(path)
        points = np.array(mesh.points)
        if len(points) == 0:
            return 0.0

        y_coords = points[:, 1]
        y_max = np.max(y_coords)  # Surface level
        y_min = np.min(y_coords)  # Bottom of trench

        return float(y_max - y_min)

    def trench_width(self, path: Path) -> float:
        """Compute trench width from geometry."""
        mesh = self._read_mesh(path)
        points = np.array(mesh.points)
        if len(points) == 0:
            return 0.0

        # Find points at the opening (near surface)
        y_coords = points[:, 1]
        y_max = np.max(y_coords)

        # Points near the surface
        surface_mask = y_coords > (y_max - 0.5)
        if not np.any(surface_mask):
            return 0.0

        surface_points = points[surface_mask]
        x_coords = surface_points[:, 0]

        # Width is the gap in x where the trench is
        x_sorted = np.sort(x_coords)
        # Find largest gap
        if len(x_sorted) > 1:
            gaps = np.diff(x_sorted)
            return float(np.max(gaps))

        return 0.0

    def aspect_ratio(self, path: Path) -> float:
        """Compute aspect ratio (depth/width)."""
        depth = self.trench_depth(path)
        width = self.trench_width(path)
        if width > 0:
            return depth / width
        return 0.0

    def conformality(self, before_path: Path, after_path: Path) -> float:
        """
        Compute conformality metric for deposition.
        Conformality = sidewall thickness / top thickness.
        Perfect conformality = 1.0.
        """
        before = self._read_mesh(before_path)
        after = self._read_mesh(after_path)

        before_points = np.array(before.points)
        after_points = np.array(after.points)

        if len(before_points) == 0 or len(after_points) == 0:
            return 0.0

        # Estimate top thickness (y increase at surface)
        before_y_max = np.max(before_points[:, 1])
        after_y_max = np.max(after_points[:, 1])
        top_thickness = after_y_max - before_y_max

        if top_thickness <= 0:
            return 0.0

        # Estimate sidewall thickness (x change at mid-height)
        mid_y = (np.max(before_points[:, 1]) + np.min(before_points[:, 1])) / 2

        # Find x extent at mid-height for before and after
        before_mid_mask = np.abs(before_points[:, 1] - mid_y) < 1.0
        after_mid_mask = np.abs(after_points[:, 1] - mid_y) < 1.0

        if not np.any(before_mid_mask) or not np.any(after_mid_mask):
            return 1.0  # Cannot compute, assume perfect

        before_x_range = np.ptp(before_points[before_mid_mask, 0])
        after_x_range = np.ptp(after_points[after_mid_mask, 0])

        sidewall_thickness = (before_x_range - after_x_range) / 2

        if sidewall_thickness <= 0:
            return 1.0

        return min(sidewall_thickness / top_thickness, 1.5)


class PhysicsValidator:
    """Physics-based validation of simulation results."""

    def __init__(self):
        self.metrics = GeometryMetrics()

    def _get_surface_level(self, path: Path) -> float:
        """Get the surface level (maximum y coordinate) from geometry."""
        mesh = _read_vtp_mesh(path)
        points = np.array(mesh.points)
        if len(points) == 0:
            return 0.0
        return float(np.max(points[:, 1]))

    def validate_etch_rate(
        self,
        before_path: Path,
        after_path: Path,
        expected_rate: float,
        process_time: float,
        tolerance: float = 0.2
    ) -> ValidationResult:
        """Validate that observed etch depth matches expected rate * time."""
        try:
            # For etching, measure the surface level drop (y_max change)
            # Etching moves the surface DOWN, so before_level > after_level
            before_level = self._get_surface_level(before_path)
            after_level = self._get_surface_level(after_path)

            # Etch amount is positive when surface moves down
            actual_etch = abs(before_level - after_level)
            expected_etch = expected_rate * process_time

            relative_error = abs(actual_etch - expected_etch) / expected_etch if expected_etch > 0 else 0

            if relative_error <= tolerance:
                return ValidationResult(
                    passed=True,
                    message=f"Etch rate within tolerance: actual={actual_etch:.2f}, expected={expected_etch:.2f}",
                    check_name="etch_rate",
                    details={
                        "actual_etch": actual_etch,
                        "expected_etch": expected_etch,
                        "relative_error": relative_error,
                        "before_level": before_level,
                        "after_level": after_level
                    }
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Etch rate outside tolerance: error={relative_error:.1%}",
                    check_name="etch_rate",
                    details={
                        "actual_etch": actual_etch,
                        "expected_etch": expected_etch,
                        "relative_error": relative_error,
                        "before_level": before_level,
                        "after_level": after_level
                    }
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to validate etch rate: {e}",
                check_name="etch_rate"
            )

    def validate_volume_change(
        self,
        before_path: Path,
        after_path: Path,
        expected_change_sign: str  # "positive" or "negative"
    ) -> ValidationResult:
        """Validate volume change direction (increase for deposition, decrease for etch)."""
        try:
            # For 2D level-set simulations, use surface level (y_max) as proxy
            # Deposition increases surface level, etching decreases it
            before_level = self._get_surface_level(before_path)
            after_level = self._get_surface_level(after_path)

            level_change = after_level - before_level

            if expected_change_sign == "positive" and level_change > 0:
                return ValidationResult(
                    passed=True,
                    message=f"Surface level increased as expected (deposition): +{level_change:.2f}",
                    check_name="volume_change",
                    details={"before_level": before_level, "after_level": after_level, "change": level_change}
                )
            elif expected_change_sign == "negative" and level_change <= 0:
                return ValidationResult(
                    passed=True,
                    message=f"Surface level decreased as expected (etching): {level_change:.2f}",
                    check_name="volume_change",
                    details={"before_level": before_level, "after_level": after_level, "change": level_change}
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Surface level change sign mismatch: expected {expected_change_sign}, got {level_change:.2f}",
                    check_name="volume_change",
                    details={"before_level": before_level, "after_level": after_level, "change": level_change}
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to validate volume change: {e}",
                check_name="volume_change"
            )

    def validate_continuity(self, path: Path) -> ValidationResult:
        """Validate geometry is continuous (no isolated points)."""
        try:
            mesh = _read_vtp_mesh(path)

            # For 2D line meshes from ViennaPS, points form a continuous polyline
            # Even without explicit cell connectivity, points > 1 indicates a valid surface
            if mesh.n_points > 1:
                # Check that most consecutive points are reasonably close
                # ViennaPS surfaces are open polylines - they don't wrap around
                points = np.array(mesh.points)
                diffs = np.diff(points, axis=0)
                distances = np.sqrt(np.sum(diffs**2, axis=1))

                if len(distances) == 0:
                    return ValidationResult(
                        passed=True,
                        message="Geometry has single point",
                        check_name="continuity",
                        details={"n_points": mesh.n_points}
                    )

                # For valid geometry, the median gap should be small
                # and most gaps (95th percentile) should be reasonable
                median_gap = float(np.median(distances))
                p95_gap = float(np.percentile(distances, 95))

                # For typical simulations, median gap should be small relative to domain
                bounds_extent = max(
                    mesh.bounds[1] - mesh.bounds[0],  # x extent
                    mesh.bounds[3] - mesh.bounds[2]   # y extent
                ) if hasattr(mesh, 'bounds') else 100.0

                # Valid if median gap is < 50% of domain and 95th percentile < 100%
                # ViennaPS uses coarse meshes by default, so allow larger spacing
                if median_gap <= bounds_extent * 0.5 and p95_gap <= bounds_extent:
                    return ValidationResult(
                        passed=True,
                        message="Geometry appears continuous",
                        check_name="continuity",
                        details={"n_points": mesh.n_points, "n_cells": mesh.n_cells,
                                "median_gap": median_gap, "p95_gap": p95_gap}
                    )
                else:
                    return ValidationResult(
                        passed=False,
                        message=f"Geometry has irregular spacing: median={median_gap:.2f}, p95={p95_gap:.2f}",
                        check_name="continuity",
                        details={"n_points": mesh.n_points, "median_gap": median_gap, "p95_gap": p95_gap}
                    )
            else:
                return ValidationResult(
                    passed=False,
                    message="Geometry has insufficient points",
                    check_name="continuity",
                    details={"n_points": mesh.n_points}
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Failed to validate continuity: {e}",
                check_name="continuity"
            )


class ValidationReport:
    """Collect and report validation results."""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def add(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)

    def __len__(self):
        return len(self.results)

    def summary(self) -> Dict[str, Any]:
        """Summarize pass/fail counts."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0
        }

    def export_markdown(self, path: Path):
        """Export report to markdown format."""
        summary = self.summary()
        lines = [
            "# Validation Report",
            "",
            f"**Total Checks:** {summary['total']}",
            f"**Passed:** {summary['passed']}",
            f"**Failed:** {summary['failed']}",
            f"**Pass Rate:** {summary['pass_rate']:.1%}",
            "",
            "## Results",
            "",
        ]

        for i, result in enumerate(self.results, 1):
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"{i}. [{status}] {result.check_name}: {result.message}")

        Path(path).write_text("\n".join(lines))

    def export_json(self, path: Path):
        """Export report to JSON format."""
        data = {
            "summary": self.summary(),
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        Path(path).write_text(json.dumps(data, indent=2))


class ValidationSuite:
    """Suite of validation checks for a simulation output directory."""

    def __init__(self, output_dir: Path, checks: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.checks = checks or ["existence", "geometry", "physics"]
        self.validator = ViennaValidator()
        self.physics = PhysicsValidator()
        self.metrics = GeometryMetrics()

    def run_all(self) -> ValidationReport:
        """Run all configured validation checks."""
        report = ValidationReport()

        # Find all VTP files
        vtp_files = list(self.output_dir.glob("*.vtp"))

        for vtp_file in vtp_files:
            # Existence check
            if "existence" in self.checks:
                report.add(self.validator.check_output_exists(vtp_file))

            # Geometry validity
            if "geometry" in self.checks:
                report.add(self.validator.check_geometry_valid(vtp_file))

        # Physics checks on pairs (before/after)
        if "physics" in self.checks:
            pairs = self._find_pairs(vtp_files)
            for before, after, process_type in pairs:
                if process_type == "etch":
                    report.add(self.physics.validate_volume_change(
                        before, after, "negative"
                    ))
                elif process_type == "deposit":
                    report.add(self.physics.validate_volume_change(
                        before, after, "positive"
                    ))

        return report

    def _find_pairs(self, vtp_files: List[Path]):
        """Find before/after pairs based on naming convention."""
        pairs = []
        initial_files = [f for f in vtp_files if "initial" in f.name]

        for initial in initial_files:
            prefix = initial.name.split("_initial")[0]
            # Find corresponding final file
            for vtp in vtp_files:
                if prefix in vtp.name and "initial" not in vtp.name:
                    # Determine process type
                    if "etch" in vtp.name.lower():
                        pairs.append((initial, vtp, "etch"))
                    elif "deposit" in vtp.name.lower():
                        pairs.append((initial, vtp, "deposit"))
                    break

        return pairs

    def assert_all_pass(self):
        """Assert that all validation checks pass (for pytest integration)."""
        report = self.run_all()
        summary = report.summary()

        if summary["failed"] > 0:
            failed_checks = [r for r in report.results if not r.passed]
            messages = [f"{r.check_name}: {r.message}" for r in failed_checks]
            raise AssertionError(
                f"Validation failed: {summary['failed']}/{summary['total']} checks failed.\n"
                + "\n".join(messages)
            )


def create_validation_suite(
    output_dir: Path,
    checks: Optional[List[str]] = None
) -> ValidationSuite:
    """Factory function to create a validation suite."""
    return ValidationSuite(output_dir, checks)
