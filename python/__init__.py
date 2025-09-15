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
        if not os.path.exists(path):
            continue

        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]


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
from viennals import IntegrationSchemeEnum as IntegrationScheme
from viennals import BoundaryConditionEnum as BoundaryType
from viennals import LogLevel as LogLevel
from . import _core as _C  # the binary inside the package

# bring d2 and d3 into the top-level namespace
d2 = _C.d2
d3 = _C.d3
_sys.modules[__name__ + ".d2"] = d2
_sys.modules[__name__ + ".d3"] = d3
ptxPath = _module_ptx_path()
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


# Config file reader helper function
def readConfigFile(fileName: str):
    """Read a config file in the ViennaPS standard config file format.

    Parameters
    ----------
    fileName: str
                Name of the config file.

    Returns
    -------
    dict
        A dictionary containing the parameters from the config file.
    """
    par_dict = {}

    with open(fileName, "r") as file:
        lines = file.readlines()
        for line in lines:

            line = line[: line.find("#")]  # remove comments

            if len(line) > 0:
                par_name = line[: line.find("=")].strip(" ")
                par_value = line[line.find("=") + 1 :]

                try:
                    val = float(par_value)
                except:
                    val = par_value

                par_dict[par_name] = val

    return par_dict


def __getattr__(name):
    # 1) common/top-level from _core
    try:
        return getattr(_C, name)
    except AttributeError as e_core:
        pass
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
