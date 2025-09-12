"""

ViennaPS
========

ViennaPS is a header-only C++ process simulation library,
which includes surface and volume representations,
a ray tracer, and physical models for the simulation of
microelectronic fabrication processes.
"""

from __future__ import annotations
import sys as sys
import sys as _sys
import viennals as ls
from viennals._core import BoundaryConditionEnum as BoundaryType
from viennals._core import IntegrationSchemeEnum as IntegrationScheme
from viennals._core import LogLevel
from viennaps._core import AdvectionParameters
from viennaps._core import CF4O2Parameters
from viennaps._core import CF4O2ParametersIons
from viennaps._core import CF4O2ParametersMask
from viennaps._core import CF4O2ParametersPassivation
from viennaps._core import CF4O2ParametersSi
from viennaps._core import CF4O2ParametersSiGe
from viennaps._core import Extrude
from viennaps._core import FaradayCageParameters
from viennaps._core import FluorocarbonParameters
from viennaps._core import FluorocarbonParametersIons
from viennaps._core import FluorocarbonParametersMask
from viennaps._core import FluorocarbonParametersPolymer
from viennaps._core import FluorocarbonParametersSi
from viennaps._core import FluorocarbonParametersSi3N4
from viennaps._core import FluorocarbonParametersSiO2
from viennaps._core import HoleShape
from viennaps._core import IBEParameters
from viennaps._core import Length
from viennaps._core import LengthUnit
from viennaps._core import Logger
from viennaps._core import Material
from viennaps._core import MaterialMap
from viennaps._core import MetaDataLevel
from viennaps._core import NormalizationType
from viennaps._core import PlasmaEtchingParameters
from viennaps._core import PlasmaEtchingParametersIons
from viennaps._core import PlasmaEtchingParametersMask
from viennaps._core import PlasmaEtchingParametersPassivation
from viennaps._core import PlasmaEtchingParametersPolymer
from viennaps._core import PlasmaEtchingParametersSubstrate
from viennaps._core import ProcessParams
from viennaps._core import RateSet
from viennaps._core import RayTracingParameters
from viennaps._core import Time
from viennaps._core import TimeUnit
from viennaps._core import constants
from viennaps._core import gpu
from viennaps._core import setNumThreads
from viennaps._core import util
from . import _core
from . import d2
from . import d3

__all__: list[str] = [
    "AdvectionParameters",
    "BoundaryType",
    "CF4O2Parameters",
    "CF4O2ParametersIons",
    "CF4O2ParametersMask",
    "CF4O2ParametersPassivation",
    "CF4O2ParametersSi",
    "CF4O2ParametersSiGe",
    "Extrude",
    "FaradayCageParameters",
    "FluorocarbonParameters",
    "FluorocarbonParametersIons",
    "FluorocarbonParametersMask",
    "FluorocarbonParametersPolymer",
    "FluorocarbonParametersSi",
    "FluorocarbonParametersSi3N4",
    "FluorocarbonParametersSiO2",
    "HoleShape",
    "IBEParameters",
    "IntegrationScheme",
    "Length",
    "LengthUnit",
    "LogLevel",
    "Logger",
    "Material",
    "MaterialMap",
    "MetaDataLevel",
    "NormalizationType",
    "PlasmaEtchingParameters",
    "PlasmaEtchingParametersIons",
    "PlasmaEtchingParametersMask",
    "PlasmaEtchingParametersPassivation",
    "PlasmaEtchingParametersPolymer",
    "PlasmaEtchingParametersSubstrate",
    "ProcessParams",
    "RateSet",
    "RayTracingParameters",
    "ReadConfigFile",
    "Time",
    "TimeUnit",
    "constants",
    "d2",
    "d3",
    "gpu",
    "ls",
    "ptxPath",
    "setNumThreads",
    "sys",
    "util",
    "version",
]

def ReadConfigFile(fileName: str):
    """
    Read a config file in the ViennaPS standard config file format.

        Parameters
        ----------
        fileName: str
                    Name of the config file.

        Returns
        -------
        dict
            A dictionary containing the parameters from the config file.

    """

def __dir__(): ...
def __getattr__(name): ...
def _module_ptx_path(): ...
def _windows_dll_path(): ...

__version__: str = "4.0.0"
ptxPath: str = ""
version: str = "4.0.0"
_C = _core
