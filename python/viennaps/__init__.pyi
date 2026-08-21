"""

ViennaPS
========

ViennaPS is a header-only C++ process simulation library,
which includes surface and volume representations,
a ray tracer, and physical models for the simulation of
microelectronic fabrication processes.
"""
from __future__ import annotations
import sys as _sys
import viennals as ls
from viennals._core import BoundaryConditionEnum as BoundaryType
from viennals._core import LogLevel
from viennals._core import SpatialSchemeEnum as SpatialScheme
from viennals._core import TemporalSchemeEnum as TemporalScheme
from viennaps._core import AdvectionParameters
from viennaps._core import AnnealMode
from viennaps._core import AtomicLayerProcessParameters
from viennaps._core import BuiltInMaterial
from viennaps._core import CF4O2Parameters
from viennaps._core import CF4O2ParametersIons
from viennaps._core import CF4O2ParametersMask
from viennaps._core import CF4O2ParametersPassivation
from viennaps._core import CF4O2ParametersSi
from viennaps._core import CF4O2ParametersSiGe
from viennaps._core import CoverageParameters
from viennaps._core import Extrude
from viennaps._core import FaradayCageParameters
from viennaps._core import FluorocarbonMaterialParameters
from viennaps._core import FluorocarbonParameters
from viennaps._core import FluorocarbonParametersIons
from viennaps._core import FluxEngineType
from viennaps._core import GpuMode
from viennaps._core import GpuPreconditioner
from viennaps._core import HoleShape
from viennaps._core import IBEParameters
from viennaps._core import IBEParametersCos4Yield
from viennaps._core import ImplantDoseControl
from viennaps._core import Length
from viennaps._core import LengthUnit
from viennaps._core import Logger
from viennaps._core import Material
from viennaps._core import MaterialCategory
from viennaps._core import MaterialInfo
from viennaps._core import MaterialKind
from viennaps._core import MaterialMap
from viennaps._core import MaterialRegistry
from viennaps._core import MaterialValueMap
from viennaps._core import MetaDataLevel
from viennaps._core import NeutralTransportParameters
from viennaps._core import NormalizationType
from viennaps._core import OxidantType
from viennaps._core import PearsonIVParameters
from viennaps._core import PlasmaEtchingParameters
from viennaps._core import PlasmaEtchingParametersIons
from viennaps._core import PlasmaEtchingParametersMask
from viennaps._core import PlasmaEtchingParametersPassivation
from viennaps._core import PlasmaEtchingParametersPolymer
from viennaps._core import PlasmaEtchingParametersSubstrate
from viennaps._core import ProcessParams
from viennaps._core import RateSet
from viennaps._core import RayTracingParameters
from viennaps._core import RenderMode
from viennaps._core import ScreenEnergyLoss
from viennaps._core import SiliconOrientation
from viennaps._core import SingleParticleALDParams
from viennaps._core import Slice
from viennaps._core import SurfaceDiffusionParameters
from viennaps._core import Time
from viennaps._core import TimeUnit
from viennaps._core import constants
from viennaps._core import getModelDbRoot
from viennaps._core import gpu
from viennaps._core import gpuAvailable
from viennaps._core import initModelDbRoot
from viennaps._core import setModelDbRoot
from viennaps._core import setNumThreads
from viennaps._core import util
from viennaps.d2 import AdvectionCallback
from viennaps.d2 import Anneal
from viennaps.d2 import BoxDistribution
from viennaps.d2 import CF4O2Etching
from viennaps.d2 import CSVFileProcess
from viennaps.d2 import CustomSphereDistribution
from viennaps.d2 import DamageTableModel
from viennaps.d2 import DenseCellSet
from viennaps.d2 import DirectionalProcess
from viennaps.d2 import Domain
from viennaps.d2 import DomainSetup
from viennaps.d2 import FaradayCageEtching
from viennaps.d2 import FluorocarbonEtching
from viennaps.d2 import GDSGeometry
from viennaps.d2 import GDSReader
from viennaps.d2 import GeometricTrenchDeposition
from viennaps.d2 import GeometryFactory
from viennaps.d2 import HBrO2Etching
from viennaps.d2 import ImplantDamageHobler
from viennaps.d2 import ImplantDualPearsonIV
from viennaps.d2 import ImplantPearsonIV
from viennaps.d2 import ImplantPearsonIVChanneling
from viennaps.d2 import ImplantProfileModel
from viennaps.d2 import ImplantTableModel
from viennaps.d2 import Interpolation
from viennaps.d2 import IonBeamEtching
from viennaps.d2 import IonImplantation
from viennaps.d2 import IsotropicProcess
from viennaps.d2 import MakeFin
from viennaps.d2 import MakeHole
from viennaps.d2 import MakePlane
from viennaps.d2 import MakeStack
from viennaps.d2 import MakeTrench
from viennaps.d2 import MultiParticleProcess
from viennaps.d2 import NetDoping
from viennaps.d2 import NeutralTransport
from viennaps.d2 import Oxidation
from viennaps.d2 import OxideRegrowth
from viennaps.d2 import Planarize
from viennaps.d2 import Process
from viennaps.d2 import ProcessModel
from viennaps.d2 import ProcessModelBase
from viennaps.d2 import RateGrid
from viennaps.d2 import Reader
from viennaps.d2 import SF6C4F8Etching
from viennaps.d2 import SF6O2Etching
from viennaps.d2 import SelectiveEpitaxy
from viennaps.d2 import SheetResistance
from viennaps.d2 import SingleParticleALD
from viennaps.d2 import SingleParticleProcess
from viennaps.d2 import SphereDistribution
from viennaps.d2 import StencilLocalLaxFriedrichsScalar
from viennaps.d2 import TEOSDeposition
from viennaps.d2 import TEOSPECVD
from viennaps.d2 import ToDiskMesh
from viennaps.d2 import VTKRenderWindow
from viennaps.d2 import WetEtching
from viennaps.d2 import Writer
from . import _core
from . import d2
from . import d3
__all__: list[str] = ['AdvectionCallback', 'AdvectionParameters', 'Anneal', 'AnnealMode', 'AtomicLayerProcessParameters', 'BoundaryType', 'BoxDistribution', 'BuiltInMaterial', 'CF4O2Etching', 'CF4O2Parameters', 'CF4O2ParametersIons', 'CF4O2ParametersMask', 'CF4O2ParametersPassivation', 'CF4O2ParametersSi', 'CF4O2ParametersSiGe', 'CSVFileProcess', 'CoverageParameters', 'CustomSphereDistribution', 'DamageTableModel', 'DenseCellSet', 'DirectionalProcess', 'Domain', 'DomainSetup', 'Extrude', 'FaradayCageEtching', 'FaradayCageParameters', 'FluorocarbonEtching', 'FluorocarbonMaterialParameters', 'FluorocarbonParameters', 'FluorocarbonParametersIons', 'FluxEngineType', 'GDSGeometry', 'GDSReader', 'GeometricTrenchDeposition', 'GeometryFactory', 'GpuMode', 'GpuPreconditioner', 'HBrO2Etching', 'HoleShape', 'IBEParameters', 'IBEParametersCos4Yield', 'ImplantDamageHobler', 'ImplantDoseControl', 'ImplantDualPearsonIV', 'ImplantPearsonIV', 'ImplantPearsonIVChanneling', 'ImplantProfileModel', 'ImplantTableModel', 'Interpolation', 'IonBeamEtching', 'IonImplantation', 'IsotropicProcess', 'Length', 'LengthUnit', 'LogLevel', 'Logger', 'MakeFin', 'MakeHole', 'MakePlane', 'MakeStack', 'MakeTrench', 'Material', 'MaterialCategory', 'MaterialInfo', 'MaterialKind', 'MaterialMap', 'MaterialRegistry', 'MaterialValueMap', 'MetaDataLevel', 'MultiParticleProcess', 'NetDoping', 'NeutralTransport', 'NeutralTransportParameters', 'NormalizationType', 'OxidantType', 'Oxidation', 'OxideRegrowth', 'PROXY_DIM', 'PearsonIVParameters', 'Planarize', 'PlasmaEtchingParameters', 'PlasmaEtchingParametersIons', 'PlasmaEtchingParametersMask', 'PlasmaEtchingParametersPassivation', 'PlasmaEtchingParametersPolymer', 'PlasmaEtchingParametersSubstrate', 'Process', 'ProcessModel', 'ProcessModelBase', 'ProcessParams', 'RateGrid', 'RateSet', 'RayTracingParameters', 'Reader', 'RenderMode', 'SF6C4F8Etching', 'SF6O2Etching', 'ScreenEnergyLoss', 'SelectiveEpitaxy', 'SheetResistance', 'SiliconOrientation', 'SingleParticleALD', 'SingleParticleALDParams', 'SingleParticleProcess', 'Slice', 'SpatialScheme', 'SphereDistribution', 'StencilLocalLaxFriedrichsScalar', 'SurfaceDiffusionParameters', 'TEOSDeposition', 'TEOSPECVD', 'TemporalScheme', 'Time', 'TimeUnit', 'ToDiskMesh', 'VTKRenderWindow', 'WetEtching', 'Writer', 'constants', 'd2', 'd3', 'getModelDbRoot', 'gpu', 'gpuAvailable', 'initModelDbRoot', 'ls', 'readConfigFile', 'setDimension', 'setModelDbRoot', 'setNumThreads', 'util', 'version']
def __dir__():
    ...
def __getattr__(name):
    ...
def _module_ptx_path():
    ...
def _windows_dll_path():
    ...
def readConfigFile(fileName: str):
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
def setDimension(d: int):
    """
    Set the dimension of the simulation (2 or 3).
    
        Parameters
        ----------
        d: int
            Dimension of the simulation (2 or 3).
        
    """
PROXY_DIM: int = 2
_SHARED_OXIDATION_TYPES: tuple = ('OxidantType', 'SiliconOrientation', 'GpuMode', 'GpuPreconditioner')
__version__: str = '4.7.0'
_name: str = 'GpuPreconditioner'
version: str = '4.7.0'
_C = _core
