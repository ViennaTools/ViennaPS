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
from viennaps3d._viennaps3d import AdvectionCallback
from viennaps3d._viennaps3d import AnisotropicProcess
from viennaps3d._viennaps3d import AtomicLayerProcess
from viennaps3d._viennaps3d import BoxDistribution
from viennaps3d._viennaps3d import DenseCellSet
from viennaps3d._viennaps3d import DirectionalEtching
from viennaps3d._viennaps3d import Domain
from viennaps3d._viennaps3d import FluorocarbonEtching
from viennaps3d._viennaps3d import FluorocarbonParameters
from viennaps3d._viennaps3d import FluorocarbonParametersIons
from viennaps3d._viennaps3d import FluorocarbonParametersMask
from viennaps3d._viennaps3d import FluorocarbonParametersPolymer
from viennaps3d._viennaps3d import FluorocarbonParametersSi
from viennaps3d._viennaps3d import FluorocarbonParametersSi3N4
from viennaps3d._viennaps3d import FluorocarbonParametersSiO2
from viennaps3d._viennaps3d import GDSGeometry
from viennaps3d._viennaps3d import GDSReader
from viennaps3d._viennaps3d import IsotropicProcess
from viennaps3d._viennaps3d import LogLevel
from viennaps3d._viennaps3d import Logger
from viennaps3d._viennaps3d import MakeFin
from viennaps3d._viennaps3d import MakeHole
from viennaps3d._viennaps3d import MakePlane
from viennaps3d._viennaps3d import MakeStack
from viennaps3d._viennaps3d import MakeTrench
from viennaps3d._viennaps3d import Material
from viennaps3d._viennaps3d import MaterialMap
from viennaps3d._viennaps3d import MeanFreePath
from viennaps3d._viennaps3d import OxideRegrowth
from viennaps3d._viennaps3d import Particle
from viennaps3d._viennaps3d import Planarize
from viennaps3d._viennaps3d import PlasmaDamage
from viennaps3d._viennaps3d import Precursor
from viennaps3d._viennaps3d import Process
from viennaps3d._viennaps3d import ProcessModel
from viennaps3d._viennaps3d import ProcessParams
from viennaps3d._viennaps3d import SF6O2Etching
from viennaps3d._viennaps3d import SF6O2Parameters
from viennaps3d._viennaps3d import SF6O2ParametersIons
from viennaps3d._viennaps3d import SF6O2ParametersMask
from viennaps3d._viennaps3d import SF6O2ParametersPassivation
from viennaps3d._viennaps3d import SF6O2ParametersSi
from viennaps3d._viennaps3d import SegmentCells
from viennaps3d._viennaps3d import SingleParticleProcess
from viennaps3d._viennaps3d import SphereDistribution
from viennaps3d._viennaps3d import TEOSDeposition
from viennaps3d._viennaps3d import Timer
from viennaps3d._viennaps3d import ToDiskMesh
from viennaps3d._viennaps3d import WriteVisualizationMesh
from viennaps3d._viennaps3d import rayReflectionConedCosine
from viennaps3d._viennaps3d import rayReflectionDiffuse
from viennaps3d._viennaps3d import rayReflectionSpecular
from viennaps3d._viennaps3d import rayTraceDirection
from viennaps3d._viennaps3d import setNumThreads
from . import _viennaps3d
__all__ = ['AdvectionCallback', 'Air', 'Al2O3', 'AnisotropicProcess', 'AtomicLayerProcess', 'BoxDistribution', 'Cu', 'D', 'DEBUG', 'DenseCellSet', 'Dielectric', 'DirectionalEtching', 'Domain', 'ERROR', 'FluorocarbonEtching', 'FluorocarbonParameters', 'FluorocarbonParametersIons', 'FluorocarbonParametersMask', 'FluorocarbonParametersPolymer', 'FluorocarbonParametersSi', 'FluorocarbonParametersSi3N4', 'FluorocarbonParametersSiO2', 'GAS', 'GDSGeometry', 'GDSReader', 'GaN', 'INFO', 'INTERMEDIATE', 'IsotropicProcess', 'LogLevel', 'Logger', 'MakeFin', 'MakeHole', 'MakePlane', 'MakeStack', 'MakeTrench', 'Mask', 'Material', 'MaterialMap', 'MeanFreePath', 'Metal', 'NEG_X', 'NEG_Y', 'NEG_Z', 'OxideRegrowth', 'POS_X', 'POS_Y', 'POS_Z', 'Particle', 'Planarize', 'PlasmaDamage', 'PolySi', 'Polymer', 'Precursor', 'Process', 'ProcessModel', 'ProcessParams', 'ReadConfigFile', 'SF6O2Etching', 'SF6O2Parameters', 'SF6O2ParametersIons', 'SF6O2ParametersMask', 'SF6O2ParametersPassivation', 'SF6O2ParametersSi', 'SegmentCells', 'Si', 'Si3N4', 'SiC', 'SiGe', 'SiN', 'SiO2', 'SiON', 'SingleParticleProcess', 'SphereDistribution', 'TEOSDeposition', 'TIMING', 'TiN', 'Timer', 'ToDiskMesh', 'Undefined', 'W', 'WARNING', 'WriteVisualizationMesh', 'rayReflectionConedCosine', 'rayReflectionDiffuse', 'rayReflectionSpecular', 'rayTraceDirection', 'setNumThreads', 'sys']
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
Air: _viennaps3d.Material  # value = <Material.Air: 17>
Al2O3: _viennaps3d.Material  # value = <Material.Al2O3: 11>
Cu: _viennaps3d.Material  # value = <Material.Cu: 13>
D: int = 3
DEBUG: _viennaps3d.LogLevel  # value = <LogLevel.DEBUG: 5>
Dielectric: _viennaps3d.Material  # value = <Material.Dielectric: 15>
ERROR: _viennaps3d.LogLevel  # value = <LogLevel.ERROR: 0>
GAS: _viennaps3d.Material  # value = <Material.GAS: 18>
GaN: _viennaps3d.Material  # value = <Material.GaN: 9>
INFO: _viennaps3d.LogLevel  # value = <LogLevel.INFO: 2>
INTERMEDIATE: _viennaps3d.LogLevel  # value = <LogLevel.INTERMEDIATE: 4>
Mask: _viennaps3d.Material  # value = <Material.Mask: 0>
Metal: _viennaps3d.Material  # value = <Material.Metal: 16>
NEG_X: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.NEG_X: 1>
NEG_Y: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.NEG_Y: 3>
NEG_Z: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.NEG_Z: 5>
POS_X: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.POS_X: 0>
POS_Y: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.POS_Y: 2>
POS_Z: _viennaps3d.rayTraceDirection  # value = <rayTraceDirection.POS_Z: 4>
PolySi: _viennaps3d.Material  # value = <Material.PolySi: 8>
Polymer: _viennaps3d.Material  # value = <Material.Polymer: 14>
Si: _viennaps3d.Material  # value = <Material.Si: 1>
Si3N4: _viennaps3d.Material  # value = <Material.Si3N4: 3>
SiC: _viennaps3d.Material  # value = <Material.SiC: 6>
SiGe: _viennaps3d.Material  # value = <Material.SiGe: 7>
SiN: _viennaps3d.Material  # value = <Material.SiN: 4>
SiO2: _viennaps3d.Material  # value = <Material.SiO2: 2>
SiON: _viennaps3d.Material  # value = <Material.SiON: 5>
TIMING: _viennaps3d.LogLevel  # value = <LogLevel.TIMING: 3>
TiN: _viennaps3d.Material  # value = <Material.TiN: 12>
Undefined: _viennaps3d.Material  # value = <Material.Undefined: -1>
W: _viennaps3d.Material  # value = <Material.W: 10>
WARNING: _viennaps3d.LogLevel  # value = <LogLevel.WARNING: 1>
