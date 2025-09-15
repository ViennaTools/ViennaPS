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


# Config file reader helper function
def ReadConfigFile(fileName: str):
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


# forward any other (common) names to _core (PEP 562)
def __getattr__(name):
    return getattr(_C, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_C)))
