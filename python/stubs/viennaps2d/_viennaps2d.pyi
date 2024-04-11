"""
ViennaPS is a header-only C++ process simulation library which includes surface and volume representations, a ray tracer, and physical models for the simulation of microelectronic fabrication processes. The main design goals are simplicity and efficiency, tailored towards scientific simulations.
"""
from __future__ import annotations
import pybind11_stubgen.typing_ext
import typing
__all__ = ['AdvectionCallback', 'Air', 'Al2O3', 'AnisotropicProcess', 'AtomicLayerProcess', 'BoxDistribution', 'Cu', 'D', 'DEBUG', 'DenseCellSet', 'Dielectric', 'DirectionalEtching', 'Domain', 'Domain3D', 'ERROR', 'Extrude', 'FluorocarbonEtching', 'FluorocarbonParameters', 'FluorocarbonParametersIons', 'FluorocarbonParametersMask', 'FluorocarbonParametersPolymer', 'FluorocarbonParametersSi', 'FluorocarbonParametersSi3N4', 'FluorocarbonParametersSiO2', 'GAS', 'GaN', 'INFO', 'INTERMEDIATE', 'IsotropicProcess', 'LogLevel', 'Logger', 'MakeFin', 'MakeHole', 'MakePlane', 'MakeStack', 'MakeTrench', 'Mask', 'Material', 'MaterialMap', 'MeanFreePath', 'Metal', 'NEG_X', 'NEG_Y', 'NEG_Z', 'OxideRegrowth', 'POS_X', 'POS_Y', 'POS_Z', 'Particle', 'Planarize', 'PlasmaDamage', 'PolySi', 'Polymer', 'Precursor', 'Process', 'ProcessModel', 'ProcessParams', 'SF6O2Etching', 'SF6O2Parameters', 'SF6O2ParametersIons', 'SF6O2ParametersMask', 'SF6O2ParametersPassivation', 'SF6O2ParametersSi', 'SegmentCells', 'Si', 'Si3N4', 'SiC', 'SiGe', 'SiN', 'SiO2', 'SiON', 'SingleParticleProcess', 'SphereDistribution', 'TEOSDeposition', 'TIMING', 'TiN', 'Timer', 'ToDiskMesh', 'Undefined', 'W', 'WARNING', 'WriteVisualizationMesh', 'rayReflectionConedCosine', 'rayReflectionDiffuse', 'rayReflectionSpecular', 'rayTraceDirection', 'setNumThreads']
class AdvectionCallback:
    def __init__(self) -> None:
        ...
    def applyPostAdvect(self, arg0: float) -> bool:
        ...
    def applyPreAdvect(self, arg0: float) -> bool:
        ...
    @property
    def domain(self) -> ...:
        ...
    @domain.setter
    def domain(*args, **kwargs):
        """
        (self: viennaps2d._viennaps2d.AdvectionCallback, arg0: psDomain<double, 2>) -> None
        """
class AnisotropicProcess(ProcessModel):
    @typing.overload
    def __init__(self, materials: list[tuple[Material, float]]) -> None:
        ...
    @typing.overload
    def __init__(self, direction100: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], direction010: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], rate100: float, rate110: float, rate111: float, rate311: float, materials: list[tuple[Material, float]]) -> None:
        ...
class AtomicLayerProcess:
    @staticmethod
    def __init__(*args, **kwargs) -> None:
        ...
    @staticmethod
    @typing.overload
    def setFirstPrecursor(*args, **kwargs) -> None:
        ...
    @staticmethod
    @typing.overload
    def setSecondPrecursor(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        ...
    @typing.overload
    def setFirstPrecursor(self, arg0: str, arg1: float, arg2: float, arg3: float, arg4: float, arg5: float) -> None:
        ...
    def setMaxLambda(self, arg0: float) -> None:
        ...
    def setMaxTimeStep(self, arg0: float) -> None:
        ...
    def setPrintInterval(self, arg0: float) -> None:
        ...
    def setPurgeParameters(self, arg0: float, arg1: float) -> None:
        ...
    def setReactionOrder(self, arg0: float) -> None:
        ...
    @typing.overload
    def setSecondPrecursor(self, arg0: str, arg1: float, arg2: float, arg3: float, arg4: float, arg5: float) -> None:
        ...
    def setStabilityFactor(self, arg0: float) -> None:
        ...
class BoxDistribution(ProcessModel):
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, halfAxes: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], gridDelta: float) -> None:
        ...
class DenseCellSet:
    def __init__(self) -> None:
        ...
    @typing.overload
    def addFillingFraction(self, arg0: int, arg1: float) -> bool:
        """
        Add to the filling fraction at given cell index.
        """
    @typing.overload
    def addFillingFraction(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], arg1: float) -> bool:
        """
        Add to the filling fraction for cell which contains given point.
        """
    def addFillingFractionInMaterial(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], arg1: float, arg2: int) -> bool:
        """
        Add to the filling fraction for cell which contains given point only if the cell has the specified material ID.
        """
    def addScalarData(self, arg0: str, arg1: float) -> None:
        """
        Add a scalar value to be stored and modified in each cell.
        """
    def buildNeighborhood(self, arg0: bool) -> None:
        """
        Generate fast neighbor access for each cell.
        """
    def clear(self) -> None:
        """
        Clear the filling fractions.
        """
    def getAverageFillingFraction(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], arg1: float) -> float:
        """
        Get the average filling at a point in some radius.
        """
    def getBoundingBox(self) -> typing.Annotated[list[typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(2)]], pybind11_stubgen.typing_ext.FixedSize(2)]:
        ...
    def getCellCenter(self, arg0: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Get the center of a cell with given index
        """
    def getCellGrid(self) -> ...:
        """
        Get the underlying mesh of the cell set.
        """
    def getCellSetPosition(self) -> bool:
        ...
    def getDepth(self) -> float:
        """
        Get the depth of the cell set.
        """
    def getElement(self, arg0: int) -> typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)]:
        """
        Get the element at the given index.
        """
    def getElements(self) -> list[typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)]]:
        """
        Get elements (cells). The indicies in the elements correspond to the corner nodes.
        """
    def getFillingFraction(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(2)]) -> float:
        """
        Get the filling fraction of the cell containing the point.
        """
    def getFillingFractions(self) -> list[float]:
        """
        Get the filling fractions of all cells.
        """
    def getGridDelta(self) -> float:
        """
        Get the cell size.
        """
    def getIndex(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]) -> int:
        """
        Get the index of the cell containing the given point.
        """
    def getNeighbors(self, arg0: int) -> typing.Annotated[list[int], pybind11_stubgen.typing_ext.FixedSize(4)]:
        """
        Get the neighbor indices for a cell.
        """
    def getNode(self, arg0: int) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        Get the node at the given index.
        """
    def getNodes(self) -> list[typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]]:
        """
        Get the nodes of the cell set which correspond to the corner points of the cells.
        """
    def getNumberOfCells(self) -> int:
        """
        Get the number of cells.
        """
    def getScalarData(self, arg0: str) -> list[float]:
        """
        Get the data stored at each cell. WARNING: This function only returns a copy of the data
        """
    def getScalarDataLabels(self) -> list[str]:
        """
        Get the labels of the scalar data stored in the cell set.
        """
    def getSurface(self) -> ...:
        """
        Get the surface level-set.
        """
    def readCellSetData(self, arg0: str) -> None:
        """
        Read cell set data from text.
        """
    def setCellSetPosition(self, arg0: bool) -> None:
        """
        Set whether the cell set should be created below (false) or above (true) the surface.
        """
    def setCoverMaterial(self, arg0: Material) -> None:
        """
        Set the material of the cells which are above or below the surface.
        """
    @typing.overload
    def setFillingFraction(self, arg0: int, arg1: float) -> bool:
        """
        Sets the filling fraction at given cell index.
        """
    @typing.overload
    def setFillingFraction(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], arg1: float) -> bool:
        """
        Sets the filling fraction for cell which contains given point.
        """
    def setPeriodicBoundary(self, arg0: typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(2)]) -> None:
        """
        Enable periodic boundary conditions in specified dimensions.
        """
    def updateMaterials(self) -> None:
        """
        Update the material IDs of the cell set. This function should be called if the level sets, the cell set is made out of, have changed. This does not work if the surface of the volume has changed. In this case, call the function 'updateSurface' first.
        """
    def updateSurface(self) -> None:
        """
        Updates the surface of the cell set. The new surface should be below the old surface as this function can only remove cells from the cell set.
        """
    def writeCellSetData(self, arg0: str) -> None:
        """
        Save cell set data in simple text format.
        """
    def writeVTU(self, arg0: str) -> None:
        """
        Write the cell set as .vtu file
        """
class DirectionalEtching(ProcessModel):
    @typing.overload
    def __init__(self, direction: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], directionalVelocity: float = 1.0, isotropicVelocity: float = 0.0, maskMaterial: Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self, direction: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], directionalVelocity: float, isotropicVelocity: float, maskMaterial: list[Material]) -> None:
        ...
class Domain:
    @staticmethod
    def applyBooleanOperation(*args, **kwargs) -> None:
        ...
    @staticmethod
    def insertNextLevelSet(*args, **kwargs) -> None:
        """
        Insert a level set to domain.
        """
    @staticmethod
    def insertNextLevelSetAsMaterial(*args, **kwargs) -> None:
        """
        Insert a level set to domain as a material.
        """
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def deepCopy(self, arg0: Domain) -> None:
        ...
    def duplicateTopLevelSet(self, arg0: Material) -> None:
        ...
    def generateCellSet(self, arg0: float, arg1: Material, arg2: bool) -> None:
        """
        Generate the cell set.
        """
    def getCellSet(self) -> ...:
        """
        Get the cell set.
        """
    def getGrid(self) -> ...:
        """
        Get the grid
        """
    def getLevelSets(self) -> list[lsDomain<double, ...] | None:
        ...
    def getMaterialMap(self) -> psMaterialMap:
        ...
    def print(self) -> None:
        ...
    def removeTopLevelSet(self) -> None:
        ...
    def saveLevelSetMesh(self, filename: str, width: int = 1) -> None:
        """
        Save the level set grids of layers in the domain.
        """
    def saveLevelSets(self, arg0: str) -> None:
        ...
    def saveSurfaceMesh(self, filename: str, addMaterialIds: bool = False) -> None:
        """
        Save the surface of the domain.
        """
    def saveVolumeMesh(self, filename: str) -> None:
        """
        Save the volume representation of the domain.
        """
    def setMaterialMap(self, arg0: psMaterialMap) -> None:
        ...
class Domain3D:
    @staticmethod
    def applyBooleanOperation(*args, **kwargs) -> None:
        ...
    @staticmethod
    def insertNextLevelSet(*args, **kwargs) -> None:
        """
        Insert a level set to domain.
        """
    @staticmethod
    def insertNextLevelSetAsMaterial(*args, **kwargs) -> None:
        """
        Insert a level set to domain as a material.
        """
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def deepCopy(self, arg0: Domain3D) -> None:
        ...
    def duplicateTopLevelSet(self, arg0: Material) -> None:
        ...
    def generateCellSet(self, position: float, coverMaterial: Material, isAboveSurface: bool) -> None:
        """
        Generate the cell set.
        """
    def getCellSet(self) -> ...:
        """
        Get the cell set.
        """
    def getGrid(self) -> ...:
        """
        Get the grid
        """
    def getLevelSets(self) -> list[lsDomain<double, ...] | None:
        ...
    def getMaterialMap(self) -> MaterialMap:
        ...
    def print(self) -> None:
        ...
    def removeTopLevelSet(self) -> None:
        ...
    def saveLevelSetMesh(self, filename: str, width: int = 1) -> None:
        """
        Save the level set grids of layers in the domain.
        """
    def saveLevelSets(self, arg0: str) -> None:
        ...
    def saveSurfaceMesh(self, filename: str, addMaterialIds: bool = True) -> None:
        """
        Save the surface of the domain.
        """
    def saveVolumeMesh(self, filename: str) -> None:
        """
        Save the volume representation of the domain.
        """
    def setMaterialMap(self, arg0: MaterialMap) -> None:
        ...
class Extrude:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, inputDomain: Domain, outputDomain: Domain3D, extent: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(2)], extrudeDimension: int, boundaryConditions: typing.Annotated[list[...], pybind11_stubgen.typing_ext.FixedSize(3)]) -> None:
        ...
    def apply(self) -> None:
        """
        Run the extrusion.
        """
    def setBoundaryConditions(self, arg0: typing.Annotated[list[...], pybind11_stubgen.typing_ext.FixedSize(3)]) -> None:
        """
        Set the boundary conditions in the extruded domain.
        """
    def setExtent(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(2)]) -> None:
        """
        Set the min and max extent in the extruded dimension.
        """
    def setExtrudeDimension(self, arg0: int) -> None:
        """
        Set which index of the added dimension (x: 0, y: 1, z: 2).
        """
    def setInputDomain(self, arg0: Domain) -> None:
        """
        Set the input domain to be extruded.
        """
    def setOutputDomain(self, arg0: Domain3D) -> None:
        """
        Set the output domain. The 3D output domain will be overwritten by the extruded domain.
        """
class FluorocarbonEtching(ProcessModel):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, ionFlux: float, etchantFlux: float, polyFlux: float, meanIonEnergy: float = 100.0, sigmaIonEnergy: float = 10.0, ionExponent: float = 100.0, deltaP: float = 0.0, etchStopDepth: float = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self, parameters: FluorocarbonParameters) -> None:
        ...
    def getParameters(self) -> FluorocarbonParameters:
        ...
    def setParameters(self, arg0: FluorocarbonParameters) -> None:
        ...
class FluorocarbonParameters:
    Ions: FluorocarbonParametersIons
    Mask: FluorocarbonParametersMask
    Polymer: FluorocarbonParametersPolymer
    Si: FluorocarbonParametersSi
    Si3N4: FluorocarbonParametersSi3N4
    SiO2: FluorocarbonParametersSiO2
    delta_p: float
    etchStopDepth: float
    etchantFlux: float
    ionFlux: float
    polyFlux: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersIons:
    exponent: float
    inflectAngle: float
    meanEnergy: float
    minAngle: float
    n_l: float
    sigmaEnergy: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersMask:
    A_sp: float
    B_sp: float
    Eth_sp: float
    beta_e: float
    beta_p: float
    rho: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersPolymer:
    A_ie: float
    Eth_ie: float
    rho: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersSi:
    A_ie: float
    A_sp: float
    B_sp: float
    E_a: float
    Eth_ie: float
    Eth_sp: float
    K: float
    rho: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersSi3N4:
    A_ie: float
    A_sp: float
    B_sp: float
    E_a: float
    Eth_ie: float
    Eth_sp: float
    K: float
    rho: float
    def __init__(self) -> None:
        ...
class FluorocarbonParametersSiO2:
    A_ie: float
    A_sp: float
    B_sp: float
    E_a: float
    Eth_ie: float
    Eth_sp: float
    K: float
    rho: float
    def __init__(self) -> None:
        ...
class IsotropicProcess(ProcessModel):
    @typing.overload
    def __init__(self, rate: float = 1.0, maskMaterial: Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self, rate: float, maskMaterial: list[Material]) -> None:
        ...
class LogLevel:
    """
    Members:
    
      ERROR
    
      WARNING
    
      INFO
    
      TIMING
    
      INTERMEDIATE
    
      DEBUG
    """
    DEBUG: typing.ClassVar[LogLevel]  # value = <LogLevel.DEBUG: 5>
    ERROR: typing.ClassVar[LogLevel]  # value = <LogLevel.ERROR: 0>
    INFO: typing.ClassVar[LogLevel]  # value = <LogLevel.INFO: 2>
    INTERMEDIATE: typing.ClassVar[LogLevel]  # value = <LogLevel.INTERMEDIATE: 4>
    TIMING: typing.ClassVar[LogLevel]  # value = <LogLevel.TIMING: 3>
    WARNING: typing.ClassVar[LogLevel]  # value = <LogLevel.WARNING: 1>
    __members__: typing.ClassVar[dict[str, LogLevel]]  # value = {'ERROR': <LogLevel.ERROR: 0>, 'WARNING': <LogLevel.WARNING: 1>, 'INFO': <LogLevel.INFO: 2>, 'TIMING': <LogLevel.TIMING: 3>, 'INTERMEDIATE': <LogLevel.INTERMEDIATE: 4>, 'DEBUG': <LogLevel.DEBUG: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Logger:
    @staticmethod
    def getInstance() -> Logger:
        ...
    @staticmethod
    def getLogLevel() -> int:
        ...
    @staticmethod
    def setLogLevel(arg0: LogLevel) -> None:
        ...
    def addDebug(self, arg0: str) -> Logger:
        ...
    def addError(self, s: str, shouldAbort: bool = True) -> Logger:
        ...
    def addInfo(self, arg0: str) -> Logger:
        ...
    @typing.overload
    def addTiming(self, arg0: str, arg1: float) -> Logger:
        ...
    @typing.overload
    def addTiming(self, arg0: str, arg1: float, arg2: float) -> Logger:
        ...
    def addWarning(self, arg0: str) -> Logger:
        ...
    def print(self) -> None:
        ...
class MakeFin:
    @staticmethod
    def __init__(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        """
        Create a fin geometry.
        """
class MakeHole:
    @staticmethod
    def __init__(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        """
        Create a hole geometry.
        """
class MakePlane:
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        """
        Create a plane geometry or add plane to existing geometry.
        """
class MakeStack:
    @staticmethod
    def __init__(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        """
        Create a stack of alternating SiO2 and Si3N4 layers.
        """
    def getHeight(self) -> float:
        """
        Returns the total height of the stack.
        """
    def getTopLayer(self) -> int:
        """
        Returns the number of layers included in the stack
        """
class MakeTrench:
    @staticmethod
    def __init__(*args, **kwargs) -> None:
        ...
    def apply(self) -> None:
        """
        Create a trench geometry.
        """
class Material:
    """
    Members:
    
      Undefined
    
      Mask
    
      Si
    
      SiO2
    
      Si3N4
    
      SiN
    
      SiON
    
      SiC
    
      SiGe
    
      PolySi
    
      GaN
    
      W
    
      Al2O3
    
      TiN
    
      Cu
    
      Polymer
    
      Dielectric
    
      Metal
    
      Air
    
      GAS
    """
    Air: typing.ClassVar[Material]  # value = <Material.Air: 17>
    Al2O3: typing.ClassVar[Material]  # value = <Material.Al2O3: 11>
    Cu: typing.ClassVar[Material]  # value = <Material.Cu: 13>
    Dielectric: typing.ClassVar[Material]  # value = <Material.Dielectric: 15>
    GAS: typing.ClassVar[Material]  # value = <Material.GAS: 18>
    GaN: typing.ClassVar[Material]  # value = <Material.GaN: 9>
    Mask: typing.ClassVar[Material]  # value = <Material.Mask: 0>
    Metal: typing.ClassVar[Material]  # value = <Material.Metal: 16>
    PolySi: typing.ClassVar[Material]  # value = <Material.PolySi: 8>
    Polymer: typing.ClassVar[Material]  # value = <Material.Polymer: 14>
    Si: typing.ClassVar[Material]  # value = <Material.Si: 1>
    Si3N4: typing.ClassVar[Material]  # value = <Material.Si3N4: 3>
    SiC: typing.ClassVar[Material]  # value = <Material.SiC: 6>
    SiGe: typing.ClassVar[Material]  # value = <Material.SiGe: 7>
    SiN: typing.ClassVar[Material]  # value = <Material.SiN: 4>
    SiO2: typing.ClassVar[Material]  # value = <Material.SiO2: 2>
    SiON: typing.ClassVar[Material]  # value = <Material.SiON: 5>
    TiN: typing.ClassVar[Material]  # value = <Material.TiN: 12>
    Undefined: typing.ClassVar[Material]  # value = <Material.Undefined: -1>
    W: typing.ClassVar[Material]  # value = <Material.W: 10>
    __members__: typing.ClassVar[dict[str, Material]]  # value = {'Undefined': <Material.Undefined: -1>, 'Mask': <Material.Mask: 0>, 'Si': <Material.Si: 1>, 'SiO2': <Material.SiO2: 2>, 'Si3N4': <Material.Si3N4: 3>, 'SiN': <Material.SiN: 4>, 'SiON': <Material.SiON: 5>, 'SiC': <Material.SiC: 6>, 'SiGe': <Material.SiGe: 7>, 'PolySi': <Material.PolySi: 8>, 'GaN': <Material.GaN: 9>, 'W': <Material.W: 10>, 'Al2O3': <Material.Al2O3: 11>, 'TiN': <Material.TiN: 12>, 'Cu': <Material.Cu: 13>, 'Polymer': <Material.Polymer: 14>, 'Dielectric': <Material.Dielectric: 15>, 'Metal': <Material.Metal: 16>, 'Air': <Material.Air: 17>, 'GAS': <Material.GAS: 18>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MaterialMap:
    @staticmethod
    def isMaterial(arg0: float, arg1: Material) -> bool:
        ...
    @staticmethod
    def mapToMaterial(arg0: float) -> Material:
        """
        Map a float to a material.
        """
    def __init__(self) -> None:
        ...
    def getMaterialAtIdx(self, arg0: int) -> Material:
        ...
    def getMaterialMap(self) -> lsMaterialMap:
        ...
    def insertNextMaterial(self, material: Material = ...) -> None:
        ...
    def size(self) -> int:
        ...
class MeanFreePath:
    def __init__(self) -> None:
        ...
    def apply(self) -> None:
        ...
    def disableSmoothing(self) -> None:
        ...
    def enableSmoothing(self) -> None:
        ...
    def setBulkLambda(self, arg0: float) -> None:
        ...
    def setDomain(self, arg0: Domain) -> None:
        ...
    def setMaterial(self, arg0: Material) -> None:
        ...
    def setNumRaysPerCell(self, arg0: float) -> None:
        ...
    def setReflectionLimit(self, arg0: int) -> None:
        ...
    def setRngSeed(self, arg0: int) -> None:
        ...
class OxideRegrowth(ProcessModel):
    def __init__(self, nitrideEtchRate: float, oxideEtchRate: float, redepositionRate: float, redepositionThreshold: float, redepositionTimeInt: float, diffusionCoefficient: float, sinkStrength: float, scallopVelocity: float, centerVelocity: float, topHeight: float, centerWidth: float, stabilityFactor: float) -> None:
        ...
class Particle:
    @staticmethod
    def initNew(*args, **kwargs) -> None:
        ...
    @staticmethod
    def surfaceCollision(*args, **kwargs) -> None:
        ...
    @staticmethod
    def surfaceReflection(*args, **kwargs) -> tuple[float, typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]]:
        ...
    def getLocalDataLabels(self) -> list[str]:
        ...
    def getSourceDistributionPower(self) -> float:
        ...
class Planarize:
    def __init__(self, geometry: Domain, cutoffHeight: float = 0.0) -> None:
        ...
    def apply(self) -> None:
        """
        Apply the planarization.
        """
class PlasmaDamage(ProcessModel):
    def __init__(self, ionEnergy: float = 100.0, meanFreePath: float = 1.0, maskMaterial: Material = ...) -> None:
        ...
class Precursor:
    adsorptionRate: float
    desorptionRate: float
    duration: float
    inFlux: float
    meanThermalVelocity: float
    name: str
    def __init__(self) -> None:
        ...
class Process:
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @staticmethod
    def setDomain(*args, **kwargs) -> None:
        """
        Set the process domain.
        """
    @typing.overload
    def __init__(self) -> None:
        ...
    def apply(self) -> None:
        """
        Run the process.
        """
    def calculateFlux(self) -> ...:
        """
        Perform a single-pass flux calculation.
        """
    def disableFluxSmoothing(self) -> None:
        """
        Disable flux smoothing
        """
    def enableFluxSmoothing(self) -> None:
        """
        Enable flux smoothing. The flux at each surface point, calculated by the ray tracer, is averaged over the surface point neighbors.
        """
    def getProcessDuration(self) -> float:
        """
        Returns the duration of the recently run process. This duration can sometimes slightly vary from the set process duration, due to the maximum time step according to the CFL condition.
        """
    def setIntegrationScheme(self, arg0: lsIntegrationSchemeEnum) -> None:
        """
        Set the integration scheme for solving the level-set equation. Possible integration schemes are specified in lsIntegrationSchemeEnum.
        """
    def setMaxCoverageInitIterations(self, arg0: int) -> None:
        """
        Set the number of iterations to initialize the coverages.
        """
    def setNumberOfRaysPerPoint(self, arg0: int) -> None:
        """
        Set the number of rays to traced for each particle in the process. The number is per point in the process geometry.
        """
    def setPrintTimeInterval(self, arg0: float) -> None:
        """
        Sets the minimum time between printing intermediate results during the process. If this is set to a non-positive value, no intermediate results are printed.
        """
    def setProcessDuration(self, arg0: float) -> None:
        """
        Set the process duration.
        """
    def setProcessModel(self, arg0: ProcessModel) -> None:
        """
        Set the process model. This has to be a pre-configured process model.
        """
    def setSourceDirection(self, arg0: rayTraceDirection) -> None:
        """
        Set source direction of the process.
        """
    def setTimeStepRatio(self, arg0: float) -> None:
        """
        Set the CFL condition to use during advection. The CFL condition sets the maximum distance a surface can be moved during one advection step. It MUST be below 0.5 to guarantee numerical stability. Defaults to 0.4999.
        """
class ProcessModel:
    @staticmethod
    def setAdvectionCallback(*args, **kwargs) -> None:
        ...
    @staticmethod
    def setGeometricModel(*args, **kwargs) -> None:
        ...
    def __init__(self) -> None:
        ...
    def getAdvectionCallback(self) -> ...:
        ...
    def getGeometricModel(self) -> ...:
        ...
    def getParticleLogSize(self, arg0: int) -> int:
        ...
    def getParticleTypes(self) -> list[...]:
        ...
    def getPrimaryDirection(self) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)] | None:
        ...
    def getProcessName(self) -> str | None:
        ...
    def getSurfaceModel(self) -> ...:
        ...
    def getVelocityField(self) -> ...:
        ...
    def insertNextParticleType(self, arg0: ...) -> None:
        ...
    def setPrimaryDirection(self, arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]) -> None:
        ...
    def setProcessName(self, arg0: str) -> None:
        ...
    def setSurfaceModel(self, arg0: ...) -> None:
        ...
    def setVelocityField(self, arg0: ...) -> None:
        ...
class ProcessParams:
    def __init__(self) -> None:
        ...
    @typing.overload
    def getScalarData(self, arg0: int) -> float:
        ...
    @typing.overload
    def getScalarData(self, arg0: int) -> float:
        ...
    @typing.overload
    def getScalarData(self, arg0: str) -> float:
        ...
    @typing.overload
    def getScalarData(self) -> list[float]:
        ...
    @typing.overload
    def getScalarData(self) -> list[float]:
        ...
    def getScalarDataIndex(self, arg0: str) -> int:
        ...
    def getScalarDataLabel(self, arg0: int) -> str:
        ...
    def insertNextScalar(self, arg0: float, arg1: str) -> None:
        ...
class SF6O2Etching(ProcessModel):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, ionFlux: float, etchantFlux: float, oxygenFlux: float, meanIonEnergy: float = 100.0, sigmaIonEnergy: float = 10.0, ionExponent: float = 100.0, oxySputterYield: float = 3.0, etchStopDepth: float = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self, parameters: SF6O2Parameters) -> None:
        ...
    def getParameters(self) -> SF6O2Parameters:
        ...
    def setParameters(self, arg0: SF6O2Parameters) -> None:
        ...
class SF6O2Parameters:
    Ions: SF6O2ParametersIons
    Mask: SF6O2ParametersMask
    Polymer: SF6O2ParametersPassivation
    Si: SF6O2ParametersSi
    beta_F: float
    beta_O: float
    etchStopDepth: float
    etchantFlux: float
    ionFlux: float
    oxygenFlux: float
    def __init__(self) -> None:
        ...
class SF6O2ParametersIons:
    exponent: float
    inflectAngle: float
    meanEnergy: float
    minAngle: float
    n_l: float
    sigmaEnergy: float
    def __init__(self) -> None:
        ...
class SF6O2ParametersMask:
    A_sp: float
    B_sp: float
    Eth_sp: float
    beta_F: float
    beta_O: float
    rho: float
    def __init__(self) -> None:
        ...
class SF6O2ParametersPassivation:
    A_ie: float
    Eth_ie: float
    def __init__(self) -> None:
        ...
class SF6O2ParametersSi:
    A_ie: float
    A_sp: float
    B_sp: float
    Eth_ie: float
    Eth_sp: float
    beta_sigma: float
    k_sigma: float
    rho: float
    def __init__(self) -> None:
        ...
class SegmentCells:
    @typing.overload
    def __init__(self, arg0: DenseCellSet) -> None:
        ...
    @typing.overload
    def __init__(self, cellSet: DenseCellSet, cellTypeString: str = 'CellType', bulkMaterial: Material = ...) -> None:
        ...
    def apply(self) -> None:
        """
        Segment the cells into surface, material, and gas cells.
        """
    def setBulkMaterial(self, arg0: Material) -> None:
        """
        Set the bulk material in the segmenter.
        """
    def setCellSet(self, arg0: DenseCellSet) -> None:
        """
        Set the cell set in the segmenter.
        """
    def setCellTypeString(self, arg0: str) -> None:
        """
        Set the cell type string in the segmenter.
        """
class SingleParticleProcess(ProcessModel):
    @typing.overload
    def __init__(self, rate: float = 1.0, stickingProbability: float = 1.0, sourceExponent: float = 1.0, maskMaterial: Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self, rate: float, stickingProbability: float, sourceExponent: float, maskMaterials: list[Material]) -> None:
        ...
class SphereDistribution(ProcessModel):
    @staticmethod
    @typing.overload
    def __init__(*args, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, radius: float, gridDelta: float) -> None:
        ...
class TEOSDeposition(ProcessModel):
    def __init__(self, stickingProbabilityP1: float, rateP1: float, orderP1: float, stickingProbabilityP2: float = 0.0, rateP2: float = 0.0, orderP2: float = 0.0) -> None:
        ...
class Timer:
    def __init__(self) -> None:
        ...
    def finish(self) -> None:
        """
        Stop the timer.
        """
    def reset(self) -> None:
        """
        Reset the timer.
        """
    def start(self) -> None:
        """
        Start the timer.
        """
    @property
    def currentDuration(self) -> int:
        """
        Get the current duration of the timer in nanoseconds.
        """
    @property
    def totalDuration(self) -> int:
        """
        Get the total duration of the timer in nanoseconds.
        """
class ToDiskMesh:
    @typing.overload
    def __init__(self, domain: Domain, mesh: ...) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain in the mesh converter.
        """
    def setMesh(self, arg0: ...) -> None:
        """
        Set the mesh in the mesh converter
        """
class WriteVisualizationMesh:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, domain: Domain, fileName: str) -> None:
        ...
    def apply(self) -> None:
        ...
    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain in the mesh converter.
        """
    def setFileName(self, arg0: str) -> None:
        """
        Set the output file name. The file name will be appended by'_volume.vtu'.
        """
class rayTraceDirection:
    """
    Members:
    
      POS_X
    
      POS_Y
    
      POS_Z
    
      NEG_X
    
      NEG_Y
    
      NEG_Z
    """
    NEG_X: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_X: 1>
    NEG_Y: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_Y: 3>
    NEG_Z: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.NEG_Z: 5>
    POS_X: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_X: 0>
    POS_Y: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_Y: 2>
    POS_Z: typing.ClassVar[rayTraceDirection]  # value = <rayTraceDirection.POS_Z: 4>
    __members__: typing.ClassVar[dict[str, rayTraceDirection]]  # value = {'POS_X': <rayTraceDirection.POS_X: 0>, 'POS_Y': <rayTraceDirection.POS_Y: 2>, 'POS_Z': <rayTraceDirection.POS_Z: 4>, 'NEG_X': <rayTraceDirection.NEG_X: 1>, 'NEG_Y': <rayTraceDirection.NEG_Y: 3>, 'NEG_Z': <rayTraceDirection.NEG_Z: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def rayReflectionConedCosine(*args, **kwargs) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
    """
    Coned cosine reflection.
    """
def rayReflectionDiffuse(*args, **kwargs) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
    """
    Diffuse reflection.
    """
def rayReflectionSpecular(arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], arg1: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
    """
    Specular reflection,
    """
def setNumThreads(arg0: int) -> None:
    ...
Air: Material  # value = <Material.Air: 17>
Al2O3: Material  # value = <Material.Al2O3: 11>
Cu: Material  # value = <Material.Cu: 13>
D: int = 2
DEBUG: LogLevel  # value = <LogLevel.DEBUG: 5>
Dielectric: Material  # value = <Material.Dielectric: 15>
ERROR: LogLevel  # value = <LogLevel.ERROR: 0>
GAS: Material  # value = <Material.GAS: 18>
GaN: Material  # value = <Material.GaN: 9>
INFO: LogLevel  # value = <LogLevel.INFO: 2>
INTERMEDIATE: LogLevel  # value = <LogLevel.INTERMEDIATE: 4>
Mask: Material  # value = <Material.Mask: 0>
Metal: Material  # value = <Material.Metal: 16>
NEG_X: rayTraceDirection  # value = <rayTraceDirection.NEG_X: 1>
NEG_Y: rayTraceDirection  # value = <rayTraceDirection.NEG_Y: 3>
NEG_Z: rayTraceDirection  # value = <rayTraceDirection.NEG_Z: 5>
POS_X: rayTraceDirection  # value = <rayTraceDirection.POS_X: 0>
POS_Y: rayTraceDirection  # value = <rayTraceDirection.POS_Y: 2>
POS_Z: rayTraceDirection  # value = <rayTraceDirection.POS_Z: 4>
PolySi: Material  # value = <Material.PolySi: 8>
Polymer: Material  # value = <Material.Polymer: 14>
Si: Material  # value = <Material.Si: 1>
Si3N4: Material  # value = <Material.Si3N4: 3>
SiC: Material  # value = <Material.SiC: 6>
SiGe: Material  # value = <Material.SiGe: 7>
SiN: Material  # value = <Material.SiN: 4>
SiO2: Material  # value = <Material.SiO2: 2>
SiON: Material  # value = <Material.SiON: 5>
TIMING: LogLevel  # value = <LogLevel.TIMING: 3>
TiN: Material  # value = <Material.TiN: 12>
Undefined: Material  # value = <Material.Undefined: -1>
W: Material  # value = <Material.W: 10>
WARNING: LogLevel  # value = <LogLevel.WARNING: 1>
__version__: str = 'VIENNAPS_VERSION'
