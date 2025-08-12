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
import viennals3d as ls
from viennals3d.viennals3d import BoundaryConditionEnum as BoundaryType
from viennals3d.viennals3d import IntegrationSchemeEnum as IntegrationScheme
from viennaps3d.viennaps3d import AdvectionCallback
from viennaps3d.viennaps3d import AdvectionParameters
from viennaps3d.viennaps3d import AtomicLayerProcess
from viennaps3d.viennaps3d import BoxDistribution
from viennaps3d.viennaps3d import CF4O2Etching
from viennaps3d.viennaps3d import CF4O2Parameters
from viennaps3d.viennaps3d import CF4O2ParametersIons
from viennaps3d.viennaps3d import CF4O2ParametersMask
from viennaps3d.viennaps3d import CF4O2ParametersPassivation
from viennaps3d.viennaps3d import CF4O2ParametersSi
from viennaps3d.viennaps3d import CF4O2ParametersSiGe
from viennaps3d.viennaps3d import CSVFileProcess
from viennaps3d.viennaps3d import DenseCellSet
from viennaps3d.viennaps3d import DirectionalProcess
from viennaps3d.viennaps3d import Domain
from viennaps3d.viennaps3d import DomainSetup
from viennaps3d.viennaps3d import FaradayCageEtching
from viennaps3d.viennaps3d import FaradayCageParameters
from viennaps3d.viennaps3d import FluorocarbonEtching
from viennaps3d.viennaps3d import FluorocarbonParameters
from viennaps3d.viennaps3d import FluorocarbonParametersIons
from viennaps3d.viennaps3d import FluorocarbonParametersMask
from viennaps3d.viennaps3d import FluorocarbonParametersPolymer
from viennaps3d.viennaps3d import FluorocarbonParametersSi
from viennaps3d.viennaps3d import FluorocarbonParametersSi3N4
from viennaps3d.viennaps3d import FluorocarbonParametersSiO2
from viennaps3d.viennaps3d import GDSGeometry
from viennaps3d.viennaps3d import GDSReader
from viennaps3d.viennaps3d import GeometryFactory
from viennaps3d.viennaps3d import HBrO2Etching
from viennaps3d.viennaps3d import HoleShape
from viennaps3d.viennaps3d import IBEParameters
from viennaps3d.viennaps3d import Interpolation
from viennaps3d.viennaps3d import IonBeamEtching
from viennaps3d.viennaps3d import IsotropicProcess
from viennaps3d.viennaps3d import Length
from viennaps3d.viennaps3d import LengthUnit
from viennaps3d.viennaps3d import LogLevel
from viennaps3d.viennaps3d import Logger
from viennaps3d.viennaps3d import MakeFin
from viennaps3d.viennaps3d import MakeHole
from viennaps3d.viennaps3d import MakePlane
from viennaps3d.viennaps3d import MakeStack
from viennaps3d.viennaps3d import MakeTrench
from viennaps3d.viennaps3d import Material
from viennaps3d.viennaps3d import MaterialMap
from viennaps3d.viennaps3d import MetaDataLevel
from viennaps3d.viennaps3d import MultiParticleProcess
from viennaps3d.viennaps3d import NormalizationType
from viennaps3d.viennaps3d import OxideRegrowth
from viennaps3d.viennaps3d import Planarize
from viennaps3d.viennaps3d import PlasmaEtchingParameters
from viennaps3d.viennaps3d import PlasmaEtchingParametersIons
from viennaps3d.viennaps3d import PlasmaEtchingParametersMask
from viennaps3d.viennaps3d import PlasmaEtchingParametersPassivation
from viennaps3d.viennaps3d import PlasmaEtchingParametersPolymer
from viennaps3d.viennaps3d import PlasmaEtchingParametersSubstrate
from viennaps3d.viennaps3d import Process
from viennaps3d.viennaps3d import ProcessModel
from viennaps3d.viennaps3d import ProcessParams
from viennaps3d.viennaps3d import RateGrid
from viennaps3d.viennaps3d import RateSet
from viennaps3d.viennaps3d import RayTracingParameters
from viennaps3d.viennaps3d import Reader
from viennaps3d.viennaps3d import SF6C4F8Etching
from viennaps3d.viennaps3d import SF6O2Etching
from viennaps3d.viennaps3d import SelectiveEpitaxy
from viennaps3d.viennaps3d import SingleParticleALD
from viennaps3d.viennaps3d import SingleParticleProcess
from viennaps3d.viennaps3d import SphereDistribution
from viennaps3d.viennaps3d import StencilLocalLaxFriedrichsScalar
from viennaps3d.viennaps3d import TEOSDeposition
from viennaps3d.viennaps3d import TEOSPECVD
from viennaps3d.viennaps3d import Time
from viennaps3d.viennaps3d import TimeUnit
from viennaps3d.viennaps3d import ToDiskMesh
from viennaps3d.viennaps3d import WetEtching
from viennaps3d.viennaps3d import Writer
from viennaps3d.viennaps3d import constants
from viennaps3d.viennaps3d import gpu
from viennaps3d.viennaps3d import ray
from viennaps3d.viennaps3d import setNumThreads
from viennaps3d.viennaps3d import util
from . import viennaps3d

__all__: list[str] = [
    "AdvectionCallback",
    "AdvectionParameters",
    "AtomicLayerProcess",
    "BoundaryType",
    "BoxDistribution",
    "CF4O2Etching",
    "CF4O2Parameters",
    "CF4O2ParametersIons",
    "CF4O2ParametersMask",
    "CF4O2ParametersPassivation",
    "CF4O2ParametersSi",
    "CF4O2ParametersSiGe",
    "CSVFileProcess",
    "D",
    "DenseCellSet",
    "DirectionalProcess",
    "Domain",
    "DomainSetup",
    "FaradayCageEtching",
    "FaradayCageParameters",
    "FluorocarbonEtching",
    "FluorocarbonParameters",
    "FluorocarbonParametersIons",
    "FluorocarbonParametersMask",
    "FluorocarbonParametersPolymer",
    "FluorocarbonParametersSi",
    "FluorocarbonParametersSi3N4",
    "FluorocarbonParametersSiO2",
    "GDSGeometry",
    "GDSReader",
    "GeometryFactory",
    "HBrO2Etching",
    "HoleShape",
    "IBEParameters",
    "IntegrationScheme",
    "Interpolation",
    "IonBeamEtching",
    "IsotropicProcess",
    "Length",
    "LengthUnit",
    "LogLevel",
    "Logger",
    "MakeFin",
    "MakeHole",
    "MakePlane",
    "MakeStack",
    "MakeTrench",
    "Material",
    "MaterialMap",
    "MetaDataLevel",
    "MultiParticleProcess",
    "NormalizationType",
    "OxideRegrowth",
    "Planarize",
    "PlasmaEtchingParameters",
    "PlasmaEtchingParametersIons",
    "PlasmaEtchingParametersMask",
    "PlasmaEtchingParametersPassivation",
    "PlasmaEtchingParametersPolymer",
    "PlasmaEtchingParametersSubstrate",
    "Process",
    "ProcessModel",
    "ProcessParams",
    "RateGrid",
    "RateSet",
    "RayTracingParameters",
    "ReadConfigFile",
    "Reader",
    "SF6C4F8Etching",
    "SF6O2Etching",
    "SelectiveEpitaxy",
    "SingleParticleALD",
    "SingleParticleProcess",
    "SphereDistribution",
    "StencilLocalLaxFriedrichsScalar",
    "TEOSDeposition",
    "TEOSPECVD",
    "Time",
    "TimeUnit",
    "ToDiskMesh",
    "WetEtching",
    "Writer",
    "constants",
    "gpu",
    "ls",
    "ptxPath",
    "ray",
    "setNumThreads",
    "sys",
    "util",
    "version",
    "viennaps3d",
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

def _module_ptx_path(): ...
def _windows_dll_path(): ...

D: int = 3
ptxPath: str = ""
version: str = '"3.7.2"'
