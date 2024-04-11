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
from viennaps2d._viennaps2d import AdvectionCallback
from viennaps2d._viennaps2d import AnisotropicProcess
from viennaps2d._viennaps2d import AtomicLayerProcess
from viennaps2d._viennaps2d import BoxDistribution
from viennaps2d._viennaps2d import DenseCellSet
from viennaps2d._viennaps2d import DirectionalEtching
from viennaps2d._viennaps2d import Domain
from viennaps2d._viennaps2d import Domain3D
from viennaps2d._viennaps2d import Extrude
from viennaps2d._viennaps2d import FluorocarbonEtching
from viennaps2d._viennaps2d import FluorocarbonParameters
from viennaps2d._viennaps2d import FluorocarbonParametersIons
from viennaps2d._viennaps2d import FluorocarbonParametersMask
from viennaps2d._viennaps2d import FluorocarbonParametersPolymer
from viennaps2d._viennaps2d import FluorocarbonParametersSi
from viennaps2d._viennaps2d import FluorocarbonParametersSi3N4
from viennaps2d._viennaps2d import FluorocarbonParametersSiO2
from viennaps2d._viennaps2d import IsotropicProcess
from viennaps2d._viennaps2d import LogLevel
from viennaps2d._viennaps2d import Logger
from viennaps2d._viennaps2d import MakeFin
from viennaps2d._viennaps2d import MakeHole
from viennaps2d._viennaps2d import MakePlane
from viennaps2d._viennaps2d import MakeStack
from viennaps2d._viennaps2d import MakeTrench
from viennaps2d._viennaps2d import Material
from viennaps2d._viennaps2d import MaterialMap
from viennaps2d._viennaps2d import MeanFreePath
from viennaps2d._viennaps2d import OxideRegrowth
from viennaps2d._viennaps2d import Particle
from viennaps2d._viennaps2d import Planarize
from viennaps2d._viennaps2d import PlasmaDamage
from viennaps2d._viennaps2d import Precursor
from viennaps2d._viennaps2d import Process
from viennaps2d._viennaps2d import ProcessModel
from viennaps2d._viennaps2d import ProcessParams
from viennaps2d._viennaps2d import SF6O2Etching
from viennaps2d._viennaps2d import SF6O2Parameters
from viennaps2d._viennaps2d import SF6O2ParametersIons
from viennaps2d._viennaps2d import SF6O2ParametersMask
from viennaps2d._viennaps2d import SF6O2ParametersPassivation
from viennaps2d._viennaps2d import SF6O2ParametersSi
from viennaps2d._viennaps2d import SegmentCells
from viennaps2d._viennaps2d import SingleParticleProcess
from viennaps2d._viennaps2d import SphereDistribution
from viennaps2d._viennaps2d import TEOSDeposition
from viennaps2d._viennaps2d import Timer
from viennaps2d._viennaps2d import ToDiskMesh
from viennaps2d._viennaps2d import WriteVisualizationMesh
from viennaps2d._viennaps2d import rayReflectionConedCosine
from viennaps2d._viennaps2d import rayReflectionDiffuse
from viennaps2d._viennaps2d import rayReflectionSpecular
from viennaps2d._viennaps2d import rayTraceDirection
from viennaps2d._viennaps2d import setNumThreads
from . import _viennaps2d
__all__ = ['AdvectionCallback', 'Air', 'Al2O3', 'AnisotropicProcess', 'AtomicLayerProcess', 'BoxDistribution', 'Cu', 'D', 'DEBUG', 'DenseCellSet', 'Dielectric', 'DirectionalEtching', 'Domain', 'Domain3D', 'ERROR', 'Extrude', 'FluorocarbonEtching', 'FluorocarbonParameters', 'FluorocarbonParametersIons', 'FluorocarbonParametersMask', 'FluorocarbonParametersPolymer', 'FluorocarbonParametersSi', 'FluorocarbonParametersSi3N4', 'FluorocarbonParametersSiO2', 'GAS', 'GaN', 'INFO', 'INTERMEDIATE', 'IsotropicProcess', 'LogLevel', 'Logger', 'MakeFin', 'MakeHole', 'MakePlane', 'MakeStack', 'MakeTrench', 'Mask', 'Material', 'MaterialMap', 'MeanFreePath', 'Metal', 'NEG_X', 'NEG_Y', 'NEG_Z', 'OxideRegrowth', 'POS_X', 'POS_Y', 'POS_Z', 'Particle', 'Planarize', 'PlasmaDamage', 'PolySi', 'Polymer', 'Precursor', 'Process', 'ProcessModel', 'ProcessParams', 'ReadConfigFile', 'SF6O2Etching', 'SF6O2Parameters', 'SF6O2ParametersIons', 'SF6O2ParametersMask', 'SF6O2ParametersPassivation', 'SF6O2ParametersSi', 'SegmentCells', 'Si', 'Si3N4', 'SiC', 'SiGe', 'SiN', 'SiO2', 'SiON', 'SingleParticleProcess', 'SphereDistribution', 'TEOSDeposition', 'TIMING', 'TiN', 'Timer', 'ToDiskMesh', 'Undefined', 'W', 'WARNING', 'WriteVisualizationMesh', 'rayReflectionConedCosine', 'rayReflectionDiffuse', 'rayReflectionSpecular', 'rayTraceDirection', 'setNumThreads', 'sys']
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
def _windows_dll_path():
    ...
Air: _viennaps2d.Material  # value = <Material.Air: 17>
Al2O3: _viennaps2d.Material  # value = <Material.Al2O3: 11>
Cu: _viennaps2d.Material  # value = <Material.Cu: 13>
D: int = 2
DEBUG: _viennaps2d.LogLevel  # value = <LogLevel.DEBUG: 5>
Dielectric: _viennaps2d.Material  # value = <Material.Dielectric: 15>
ERROR: _viennaps2d.LogLevel  # value = <LogLevel.ERROR: 0>
GAS: _viennaps2d.Material  # value = <Material.GAS: 18>
GaN: _viennaps2d.Material  # value = <Material.GaN: 9>
INFO: _viennaps2d.LogLevel  # value = <LogLevel.INFO: 2>
INTERMEDIATE: _viennaps2d.LogLevel  # value = <LogLevel.INTERMEDIATE: 4>
Mask: _viennaps2d.Material  # value = <Material.Mask: 0>
Metal: _viennaps2d.Material  # value = <Material.Metal: 16>
NEG_X: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.NEG_X: 1>
NEG_Y: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.NEG_Y: 3>
NEG_Z: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.NEG_Z: 5>
POS_X: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.POS_X: 0>
POS_Y: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.POS_Y: 2>
POS_Z: _viennaps2d.rayTraceDirection  # value = <rayTraceDirection.POS_Z: 4>
PolySi: _viennaps2d.Material  # value = <Material.PolySi: 8>
Polymer: _viennaps2d.Material  # value = <Material.Polymer: 14>
Si: _viennaps2d.Material  # value = <Material.Si: 1>
Si3N4: _viennaps2d.Material  # value = <Material.Si3N4: 3>
SiC: _viennaps2d.Material  # value = <Material.SiC: 6>
SiGe: _viennaps2d.Material  # value = <Material.SiGe: 7>
SiN: _viennaps2d.Material  # value = <Material.SiN: 4>
SiO2: _viennaps2d.Material  # value = <Material.SiO2: 2>
SiON: _viennaps2d.Material  # value = <Material.SiON: 5>
TIMING: _viennaps2d.LogLevel  # value = <LogLevel.TIMING: 3>
TiN: _viennaps2d.Material  # value = <Material.TiN: 12>
Undefined: _viennaps2d.Material  # value = <Material.Undefined: -1>
W: _viennaps2d.Material  # value = <Material.W: 10>
WARNING: _viennaps2d.LogLevel  # value = <LogLevel.WARNING: 1>
