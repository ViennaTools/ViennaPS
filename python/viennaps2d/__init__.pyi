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
import viennals2d as ls
from viennals2d.viennals2d import BoundaryConditionEnum as BoundaryType
from viennals2d.viennals2d import IntegrationSchemeEnum as IntegrationScheme
from viennaps2d.viennaps2d import AdvectionCallback
from viennaps2d.viennaps2d import AdvectionParameters
from viennaps2d.viennaps2d import AtomicLayerProcess
from viennaps2d.viennaps2d import BoxDistribution
from viennaps2d.viennaps2d import CF4O2Etching
from viennaps2d.viennaps2d import CF4O2Parameters
from viennaps2d.viennaps2d import CF4O2ParametersIons
from viennaps2d.viennaps2d import CF4O2ParametersMask
from viennaps2d.viennaps2d import CF4O2ParametersPassivation
from viennaps2d.viennaps2d import CF4O2ParametersSi
from viennaps2d.viennaps2d import CF4O2ParametersSiGe
from viennaps2d.viennaps2d import CSVFileProcess
from viennaps2d.viennaps2d import DenseCellSet
from viennaps2d.viennaps2d import DirectionalProcess
from viennaps2d.viennaps2d import Domain
from viennaps2d.viennaps2d import Domain3D
from viennaps2d.viennaps2d import DomainSetup
from viennaps2d.viennaps2d import Extrude
from viennaps2d.viennaps2d import FaradayCageEtching
from viennaps2d.viennaps2d import FaradayCageParameters
from viennaps2d.viennaps2d import FluorocarbonEtching
from viennaps2d.viennaps2d import FluorocarbonParameters
from viennaps2d.viennaps2d import FluorocarbonParametersIons
from viennaps2d.viennaps2d import FluorocarbonParametersMask
from viennaps2d.viennaps2d import FluorocarbonParametersPolymer
from viennaps2d.viennaps2d import FluorocarbonParametersSi
from viennaps2d.viennaps2d import FluorocarbonParametersSi3N4
from viennaps2d.viennaps2d import FluorocarbonParametersSiO2
from viennaps2d.viennaps2d import GDSGeometry
from viennaps2d.viennaps2d import GDSReader
from viennaps2d.viennaps2d import GeometryFactory
from viennaps2d.viennaps2d import HBrO2Etching
from viennaps2d.viennaps2d import HoleShape
from viennaps2d.viennaps2d import IBEParameters
from viennaps2d.viennaps2d import Interpolation
from viennaps2d.viennaps2d import IonBeamEtching
from viennaps2d.viennaps2d import IsotropicProcess
from viennaps2d.viennaps2d import Length
from viennaps2d.viennaps2d import LengthUnit
from viennaps2d.viennaps2d import LogLevel
from viennaps2d.viennaps2d import Logger
from viennaps2d.viennaps2d import MakeFin
from viennaps2d.viennaps2d import MakeHole
from viennaps2d.viennaps2d import MakePlane
from viennaps2d.viennaps2d import MakeStack
from viennaps2d.viennaps2d import MakeTrench
from viennaps2d.viennaps2d import Material
from viennaps2d.viennaps2d import MaterialMap
from viennaps2d.viennaps2d import MetaDataLevel
from viennaps2d.viennaps2d import MultiParticleProcess
from viennaps2d.viennaps2d import NormalizationType
from viennaps2d.viennaps2d import OxideRegrowth
from viennaps2d.viennaps2d import Planarize
from viennaps2d.viennaps2d import PlasmaEtchingParameters
from viennaps2d.viennaps2d import PlasmaEtchingParametersIons
from viennaps2d.viennaps2d import PlasmaEtchingParametersMask
from viennaps2d.viennaps2d import PlasmaEtchingParametersPassivation
from viennaps2d.viennaps2d import PlasmaEtchingParametersPolymer
from viennaps2d.viennaps2d import PlasmaEtchingParametersSubstrate
from viennaps2d.viennaps2d import Process
from viennaps2d.viennaps2d import ProcessModel
from viennaps2d.viennaps2d import ProcessParams
from viennaps2d.viennaps2d import RateGrid
from viennaps2d.viennaps2d import RateSet
from viennaps2d.viennaps2d import RayTracingParameters
from viennaps2d.viennaps2d import Reader
from viennaps2d.viennaps2d import SF6C4F8Etching
from viennaps2d.viennaps2d import SF6O2Etching
from viennaps2d.viennaps2d import SelectiveEpitaxy
from viennaps2d.viennaps2d import SingleParticleALD
from viennaps2d.viennaps2d import SingleParticleProcess
from viennaps2d.viennaps2d import SphereDistribution
from viennaps2d.viennaps2d import StencilLocalLaxFriedrichsScalar
from viennaps2d.viennaps2d import TEOSDeposition
from viennaps2d.viennaps2d import TEOSPECVD
from viennaps2d.viennaps2d import Time
from viennaps2d.viennaps2d import TimeUnit
from viennaps2d.viennaps2d import ToDiskMesh
from viennaps2d.viennaps2d import WetEtching
from viennaps2d.viennaps2d import Writer
from viennaps2d.viennaps2d import constants
from viennaps2d.viennaps2d import ray
from viennaps2d.viennaps2d import setNumThreads
from viennaps2d.viennaps2d import util
from . import viennaps2d

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
    "Domain3D",
    "DomainSetup",
    "Extrude",
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
    "ls",
    "ray",
    "setNumThreads",
    "sys",
    "util",
    "version",
    "viennaps2d",
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

D: int = 2
version: str = '"3.7.2"'
