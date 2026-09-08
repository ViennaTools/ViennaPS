"""
ViennaPS
========

ViennaPS is a header-only C++ process simulation library,
which includes surface and volume representations,
a ray tracer, and physical models for the simulation of
microelectronic fabrication processes.
"""


def _windows_dll_path():
    import os

    additional_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "viennaps.libs")
    ]

    for path in additional_paths:
        if os.path.isdir(path):
            os.add_dll_directory(path)
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        else:
            print(f"Warning: DLL path {path} does not exist.")


def _module_ptx_path():
    from importlib.util import find_spec
    import os

    spec = find_spec("viennaps")
    install_path = os.path.dirname(os.path.abspath(spec.origin))
    return os.path.join(install_path, "ptx")


import sys as _sys

if _sys.platform == "win32":
    _windows_dll_path()


import viennals as ls

# Convenience imports
from viennals import SpatialSchemeEnum as SpatialScheme
from viennals import TemporalSchemeEnum as TemporalScheme
from viennals import BoundaryConditionEnum as BoundaryType
from viennals import Domain as LevelSet

from viennals import MakeGeometry as MakeGeometry
from viennals import Plane as Plane
from viennals import Sphere as Sphere
from viennals import Cylinder as Cylinder
from viennals import Box as Box

from viennals import BooleanOperationEnum as BooleanOperationType
from viennals import BooleanOperation as BooleanOperation

from viennals import VTKWriter as VTKWriter
from viennals import LogLevel as LogLevel
from . import _core as _C  # the binary inside the package

# bring d2 and d3 into the top-level namespace
d2 = _C.d2
d3 = _C.d3
_sys.modules[__name__ + ".d2"] = d2
_sys.modules[__name__ + ".d3"] = d3

_SHARED_OXIDATION_TYPES = (
    "OxidantType",
    "SiliconOrientation",
    "GpuMode",
    "GpuPreconditioner",
)
for _name in _SHARED_OXIDATION_TYPES:
    if hasattr(_C, _name):
        setattr(d2, _name, getattr(_C, _name))
        setattr(d3, _name, getattr(_C, _name))

PROXY_DIM = 2  # default dimension is 2D


def setDimension(d: int):
    """Set the dimension of the simulation (2 or 3).

    Parameters
    ----------
    d: int
        Dimension of the simulation (2 or 3).
    """
    global PROXY_DIM
    if d == 2 or d == 3:
        PROXY_DIM = d
        ls.setDimension(d)
    else:
        raise ValueError("Dimension must be 2 or 3.")


def readConfigFile(fileName: str) -> dict:
    """Read a config file in the ViennaPS standard config file format.

    Parameters
    ----------
    fileName : str
        Name of the config file.

    Returns
    -------
    dict
        A dictionary containing the parameters from the config file.
        Numeric values are returned as floats, and comma-separated numeric
        values as lists of floats. Other values are returned as strings or
        lists of strings, with surrounding whitespace removed. If any list
        item is nonnumeric, all items in that list are returned as strings.
        Comments starting with '#' and lines without '=' are ignored.
    """
    par_dict = {}

    with open(fileName, "r", encoding="utf-8") as file:
        for line in file:
            line = line.split("#", 1)[0]
            if "=" not in line:
                continue

            par_name, par_value = line.split("=", 1)
            par_name = par_name.strip()
            par_value = par_value.strip()

            try:
                if "," in par_value:
                    val = [float(value) for value in par_value.split(",")]
                else:
                    val = float(par_value)
            except ValueError:
                if "," in par_value:
                    val = [value.strip() for value in par_value.split(",")]
                else:
                    val = par_value

            par_dict[par_name] = val

    return par_dict


def __getattr__(name):
    # 1) common/top-level from _core
    e_core = None
    try:
        return getattr(_C, name)
    except AttributeError as e:
        e_core = e
    # 2) fallback to current default dimension
    m = d2 if PROXY_DIM == 2 else d3
    try:
        return getattr(m, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from e_core


def __dir__():
    return sorted(set(globals()) | set(dir(_C)) | set(dir(d2)) | set(dir(d3)))
