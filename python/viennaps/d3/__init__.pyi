"""
3D bindings
"""

from __future__ import annotations
import collections.abc
import enum
import typing
import viennals._core
import viennals.d3
import viennaps._core
from . import gpu

__all__: list[str] = [
    "AdvectionCallback",
    "BoxDistribution",
    "CF4O2Etching",
    "CSVFileProcess",
    "DenseCellSet",
    "DirectionalProcess",
    "Domain",
    "DomainSetup",
    "FaradayCageEtching",
    "FluorocarbonEtching",
    "GDSGeometry",
    "GDSReader",
    "GeometricTrenchDeposition",
    "GeometryFactory",
    "HBrO2Etching",
    "Interpolation",
    "IonBeamEtching",
    "IsotropicProcess",
    "MakeFin",
    "MakeHole",
    "MakePlane",
    "MakeStack",
    "MakeTrench",
    "MultiParticleProcess",
    "OxideRegrowth",
    "Planarize",
    "Process",
    "ProcessModel",
    "ProcessModelBase",
    "RateGrid",
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
    "ToDiskMesh",
    "WetEtching",
    "Writer",
    "gpu",
]

class AdvectionCallback:
    domain: Domain
    def __init__(self) -> None: ...
    def applyPostAdvect(self, arg0: typing.SupportsFloat) -> bool: ...
    def applyPreAdvect(self, arg0: typing.SupportsFloat) -> bool: ...

class BoxDistribution(ProcessModel):
    @typing.overload
    def __init__(
        self,
        halfAxes: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        mask: viennals.d3.Domain,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        halfAxes: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...

class CF4O2Etching(ProcessModel):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        ionFlux: typing.SupportsFloat,
        etchantFlux: typing.SupportsFloat,
        oxygenFlux: typing.SupportsFloat,
        polymerFlux: typing.SupportsFloat,
        meanIonEnergy: typing.SupportsFloat = 100.0,
        sigmaIonEnergy: typing.SupportsFloat = 10.0,
        ionExponent: typing.SupportsFloat = 100.0,
        oxySputterYield: typing.SupportsFloat = 3.0,
        polySputterYield: typing.SupportsFloat = 3.0,
        etchStopDepth: typing.SupportsFloat = -1.7976931348623157e308,
    ) -> None: ...
    @typing.overload
    def __init__(self, parameters: viennaps._core.CF4O2Parameters) -> None: ...
    def getParameters(self) -> viennaps._core.CF4O2Parameters: ...
    def setParameters(self, arg0: viennaps._core.CF4O2Parameters) -> None: ...

class CSVFileProcess(ProcessModel):
    def __init__(
        self,
        ratesFile: str,
        direction: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        offset: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
        isotropicComponent: typing.SupportsFloat = 0.0,
        directionalComponent: typing.SupportsFloat = 1.0,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material] = ...,
        calculateVisibility: bool = True,
    ) -> None: ...
    def setCustomInterpolator(self, function: collections.abc.Callable) -> None: ...
    def setIDWNeighbors(self, k: typing.SupportsInt = 4) -> None: ...
    @typing.overload
    def setInterpolationMode(self, mode: Interpolation) -> None: ...
    @typing.overload
    def setInterpolationMode(self, mode: str) -> None: ...
    def setOffset(
        self,
        offset: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...

class DenseCellSet:
    def __init__(self) -> None: ...
    @typing.overload
    def addFillingFraction(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> bool:
        """
        Add to the filling fraction at given cell index.
        """

    @typing.overload
    def addFillingFraction(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> bool:
        """
        Add to the filling fraction for cell which contains given point.
        """

    def addFillingFractionInMaterial(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
        arg2: typing.SupportsInt,
    ) -> bool:
        """
        Add to the filling fraction for cell which contains given point only if the cell has the specified material ID.
        """

    def addScalarData(self, arg0: str, arg1: typing.SupportsFloat) -> None:
        """
        Add a scalar value to be stored and modified in each cell.
        """

    def buildNeighborhood(self, forceRebuild: bool = False) -> None:
        """
        Generate fast neighbor access for each cell.
        """

    def clear(self) -> None:
        """
        Clear the filling fractions.
        """

    def fromLevelSets(
        self,
        levelSets: collections.abc.Sequence[viennals.d3.Domain],
        materialMap: viennals._core.MaterialMap = None,
        depth: typing.SupportsFloat = 0.0,
    ) -> None: ...
    def getAverageFillingFraction(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> float:
        """
        Get the average filling at a point in some radius.
        """

    def getBoundingBox(
        self,
    ) -> typing.Annotated[
        list[typing.Annotated[list[float], "FixedSize(3)"]], "FixedSize(2)"
    ]: ...
    def getCellCenter(
        self, arg0: typing.SupportsInt
    ) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Get the center of a cell with given index
        """

    def getCellGrid(self) -> viennals._core.Mesh:
        """
        Get the underlying mesh of the cell set.
        """

    def getDepth(self) -> float:
        """
        Get the depth of the cell set.
        """

    def getElement(
        self, arg0: typing.SupportsInt
    ) -> typing.Annotated[list[int], "FixedSize(8)"]:
        """
        Get the element at the given index.
        """

    def getElements(self) -> list[typing.Annotated[list[int], "FixedSize(8)"]]:
        """
        Get elements (cells). The indicies in the elements correspond to the corner nodes.
        """

    def getFillingFraction(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> float:
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

    def getIndex(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> int:
        """
        Get the index of the cell containing the given point.
        """

    def getNeighbors(
        self, arg0: typing.SupportsInt
    ) -> typing.Annotated[list[int], "FixedSize(6)"]:
        """
        Get the neighbor indices for a cell.
        """

    def getNode(
        self, arg0: typing.SupportsInt
    ) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Get the node at the given index.
        """

    def getNodes(self) -> list[typing.Annotated[list[float], "FixedSize(3)"]]:
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

    def getSurface(self) -> viennals.d3.Domain:
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

    def setCoverMaterial(self, arg0: typing.SupportsInt) -> None:
        """
        Set the material of the cells which are above or below the surface.
        """

    @typing.overload
    def setFillingFraction(
        self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat
    ) -> bool:
        """
        Sets the filling fraction at given cell index.
        """

    @typing.overload
    def setFillingFraction(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        arg1: typing.SupportsFloat,
    ) -> bool:
        """
        Sets the filling fraction for cell which contains given point.
        """

    def setPeriodicBoundary(
        self, arg0: typing.Annotated[collections.abc.Sequence[bool], "FixedSize(3)"]
    ) -> None:
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

class DirectionalProcess(ProcessModel):
    @typing.overload
    def __init__(
        self,
        direction: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        directionalVelocity: typing.SupportsFloat,
        isotropicVelocity: typing.SupportsFloat = 0.0,
        maskMaterial: viennaps._core.Material = ...,
        calculateVisibility: bool = True,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        direction: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        directionalVelocity: typing.SupportsFloat,
        isotropicVelocity: typing.SupportsFloat,
        maskMaterial: collections.abc.Sequence[viennaps._core.Material],
        calculateVisibility: bool = True,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, rateSets: collections.abc.Sequence[viennaps._core.RateSet]
    ) -> None: ...
    @typing.overload
    def __init__(self, rateSet: viennaps._core.RateSet) -> None: ...

class Domain:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain) -> None:
        """
        Deep copy constructor.
        """

    @typing.overload
    def __init__(
        self,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        boundary: viennals._core.BoundaryConditionEnum = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        boundary: viennals._core.BoundaryConditionEnum = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        bounds: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
        boundaryConditions: typing.Annotated[
            collections.abc.Sequence[viennals._core.BoundaryConditionEnum],
            "FixedSize(3)",
        ],
        gridDelta: typing.SupportsFloat = 1.0,
    ) -> None: ...
    @typing.overload
    def __init__(self, setup: DomainSetup) -> None: ...
    @typing.overload
    def addMetaData(self, arg0: str, arg1: typing.SupportsFloat) -> None:
        """
        Add a single metadata entry to the domain.
        """

    @typing.overload
    def addMetaData(
        self, arg0: str, arg1: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None:
        """
        Add a single metadata entry to the domain.
        """

    @typing.overload
    def addMetaData(
        self,
        arg0: collections.abc.Mapping[
            str, collections.abc.Sequence[typing.SupportsFloat]
        ],
    ) -> None:
        """
        Add metadata to the domain.
        """

    def applyBooleanOperation(
        self,
        levelSet: viennals.d3.Domain,
        operation: viennals._core.BooleanOperationEnum,
        applyToAll: bool = True,
    ) -> None:
        """
        Apply a boolean operation with the passed Level-Set to all (or top only) Level-Sets in the domain.
        """

    def clear(self) -> None: ...
    def clearMetaData(self, clearDomainData: bool = False) -> None:
        """
        Clear meta data from domain.
        """

    def deepCopy(self, arg0: Domain) -> None: ...
    def disableMetaData(self) -> None:
        """
        Disable adding meta data to domain.
        """

    def duplicateTopLevelSet(self, arg0: viennaps._core.Material) -> None:
        """
        Duplicate the top level set. Should be used before a deposition process.
        """

    def enableMetaData(self, level: viennaps._core.MetaDataLevel = ...) -> None:
        """
        Enable adding meta data from processes to domain.
        """

    def generateCellSet(
        self, arg0: typing.SupportsFloat, arg1: viennaps._core.Material, arg2: bool
    ) -> None:
        """
        Generate the cell set.
        """

    def getBoundaryConditions(
        self,
    ) -> typing.Annotated[list[viennals._core.BoundaryConditionEnum], "FixedSize(3)"]:
        """
        Get the boundary conditions of the domain.
        """

    def getBoundingBox(
        self,
    ) -> typing.Annotated[
        list[typing.Annotated[list[float], "FixedSize(3)"]], "FixedSize(2)"
    ]:
        """
        Get the bounding box of the domain.
        """

    def getCellSet(self) -> DenseCellSet:
        """
        Get the cell set.
        """

    def getGrid(self) -> viennals.d3.hrleGrid:
        """
        Get the grid
        """

    def getGridDelta(self) -> float:
        """
        Get the grid delta.
        """

    def getLevelSetMesh(
        self, width: typing.SupportsInt = 1
    ) -> list[viennals._core.Mesh]:
        """
        Get the level set grids of layers in the domain.
        """

    def getLevelSets(self) -> list[viennals.d3.Domain]: ...
    def getMaterialMap(self) -> viennaps._core.MaterialMap: ...
    def getMetaData(self) -> dict[str, list[float]]:
        """
        Get meta data (e.g. process data) stored in the domain
        """

    def getMetaDataLevel(self) -> viennaps._core.MetaDataLevel:
        """
        Get the current meta data level of the domain.
        """

    def getSetup(self) -> DomainSetup:
        """
        Get the domain setup.
        """

    def getSurfaceMesh(
        self,
        addInterfaces: bool = False,
        wrappingLayerEpsilon: typing.SupportsFloat = 0.01,
        boolMaterials: bool = False,
    ) -> viennals._core.Mesh:
        """
        Get the surface mesh of the domain
        """

    def insertNextLevelSet(
        self, levelset: viennals.d3.Domain, wrapLowerLevelSet: bool = True
    ) -> None:
        """
        Insert a level set to domain.
        """

    def insertNextLevelSetAsMaterial(
        self,
        levelSet: viennals.d3.Domain,
        material: viennaps._core.Material,
        wrapLowerLevelSet: bool = True,
    ) -> None:
        """
        Insert a level set to domain as a material.
        """

    def print(self, hrleInfo: bool = False) -> None:
        """
        Print the domain information.
        """

    def removeLevelSet(self, arg0: typing.SupportsInt, arg1: bool) -> None: ...
    def removeMaterial(self, arg0: viennaps._core.Material) -> None: ...
    def removeTopLevelSet(self) -> None: ...
    def saveHullMesh(
        self, filename: str, wrappingLayerEpsilon: typing.SupportsFloat = 0.01
    ) -> None:
        """
        Save the hull of the domain.
        """

    def saveLevelSetMesh(self, filename: str, width: typing.SupportsInt = 1) -> None:
        """
        Save the level set grids of layers in the domain.
        """

    def saveLevelSets(self, filename: str) -> None: ...
    def saveSurfaceMesh(
        self,
        filename: str,
        addInterfaces: bool = False,
        wrappingLayerEpsilon: typing.SupportsFloat = 0.01,
        boolMaterials: bool = False,
    ) -> None:
        """
        Save the surface of the domain.
        """

    def saveVolumeMesh(
        self, filename: str, wrappingLayerEpsilon: typing.SupportsFloat = 0.01
    ) -> None:
        """
        Save the volume representation of the domain.
        """

    def setMaterialMap(self, arg0: viennaps._core.MaterialMap) -> None: ...
    @typing.overload
    def setup(self, arg0: DomainSetup) -> None:
        """
        Setup the domain.
        """

    @typing.overload
    def setup(
        self,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat = 0.0,
        boundary: viennals._core.BoundaryConditionEnum = ...,
    ) -> None:
        """
        Setup the domain.
        """

class DomainSetup:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        boundary: viennals._core.BoundaryConditionEnum = ...,
    ) -> None: ...
    def boundaryCons(
        self,
    ) -> typing.Annotated[
        list[viennals._core.BoundaryConditionEnum], "FixedSize(3)"
    ]: ...
    def bounds(self) -> typing.Annotated[list[float], "FixedSize(6)"]: ...
    def check(self) -> None: ...
    def grid(self) -> viennals.d3.hrleGrid: ...
    def gridDelta(self) -> float: ...
    def halveXAxis(self) -> None: ...
    def halveYAxis(self) -> None: ...
    def hasPeriodicBoundary(self) -> bool: ...
    def isValid(self) -> bool: ...
    def print(self) -> None: ...
    def xExtent(self) -> float: ...
    def yExtent(self) -> float: ...

class FaradayCageEtching(ProcessModel):
    def __init__(
        self,
        parameters: viennaps._core.FaradayCageParameters,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...

class FluorocarbonEtching(ProcessModel):
    def __init__(self, parameters: viennaps._core.FluorocarbonParameters) -> None: ...
    def setParameters(self, arg0: viennaps._core.FluorocarbonParameters) -> None: ...

class GDSGeometry:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, gridDelta: typing.SupportsFloat) -> None: ...
    @typing.overload
    def __init__(
        self,
        gridDelta: typing.SupportsFloat,
        boundaryConditions: typing.Annotated[
            collections.abc.Sequence[viennals._core.BoundaryConditionEnum],
            "FixedSize(3)",
        ],
    ) -> None: ...
    def addBlur(
        self,
        sigmas: collections.abc.Sequence[typing.SupportsFloat],
        weights: collections.abc.Sequence[typing.SupportsFloat],
        threshold: typing.SupportsFloat = 0.5,
        delta: typing.SupportsFloat = 0.0,
        gridRefinement: typing.SupportsInt = 4,
    ) -> None:
        """
        Set parameters for applying mask blurring.
        """

    def getAllLayers(self) -> set[int]:
        """
        Return a set of all layers found in the GDS file.
        """

    def getBounds(self) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get the bounds of the geometry.
        """

    def getNumberOfStructures(self) -> int:
        """
        Return number of structure definitions.
        """

    def layerToLevelSet(
        self,
        layer: typing.SupportsInt,
        baseHeight: typing.SupportsFloat = 0.0,
        height: typing.SupportsFloat = 1.0,
        mask: bool = False,
        blurLayer: bool = True,
    ) -> viennals.d3.Domain: ...
    def print(self) -> None:
        """
        Print the geometry contents.
        """

    def setBoundaryConditions(
        self, arg0: collections.abc.Sequence[viennals._core.BoundaryConditionEnum]
    ) -> None:
        """
        Set the boundary conditions
        """

    def setBoundaryPadding(
        self, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat
    ) -> None:
        """
        Set padding between the largest point of the geometry and the boundary of the domain.
        """

    def setGridDelta(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the grid spacing.
        """

class GDSReader:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: GDSGeometry, arg1: str) -> None: ...
    def apply(self) -> None:
        """
        Parse the GDS file.
        """

    def setFileName(self, arg0: str) -> None:
        """
        Set name of the GDS file.
        """

    def setGeometry(self, arg0: GDSGeometry) -> None:
        """
        Set the domain to be parsed in.
        """

class GeometricTrenchDeposition(ProcessModel):
    def __init__(
        self,
        trenchWidth: typing.SupportsFloat,
        trenchDepth: typing.SupportsFloat,
        depositionRate: typing.SupportsFloat,
        bottomMed: typing.SupportsFloat,
        a: typing.SupportsFloat,
        b: typing.SupportsFloat,
        n: typing.SupportsFloat,
    ) -> None: ...

class GeometryFactory:
    def __init__(
        self, domainSetup: DomainSetup, name: str = "GeometryFactory"
    ) -> None: ...
    def makeBoxStencil(
        self,
        position: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        width: typing.SupportsFloat,
        height: typing.SupportsFloat,
        angle: typing.SupportsFloat = 0.0,
        length: typing.SupportsFloat = -1.0,
    ) -> viennals.d3.Domain: ...
    def makeCylinderStencil(
        self,
        position: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        radius: typing.SupportsFloat,
        height: typing.SupportsFloat,
        angle: typing.SupportsFloat = 0.0,
    ) -> viennals.d3.Domain: ...
    def makeMask(
        self, base: typing.SupportsFloat, height: typing.SupportsFloat
    ) -> viennals.d3.Domain: ...
    def makeSubstrate(self, base: typing.SupportsFloat) -> viennals.d3.Domain: ...

class HBrO2Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        ionFlux: typing.SupportsFloat,
        etchantFlux: typing.SupportsFloat,
        oxygenFlux: typing.SupportsFloat,
        meanIonEnergy: typing.SupportsFloat = 100.0,
        sigmaIonEnergy: typing.SupportsFloat = 10.0,
        ionExponent: typing.SupportsFloat = 100.0,
        oxySputterYield: typing.SupportsFloat = 3.0,
        etchStopDepth: typing.SupportsFloat = -1.7976931348623157e308,
    ) -> None: ...
    @typing.overload
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> viennaps._core.PlasmaEtchingParameters: ...
    def setParameters(self, arg0: viennaps._core.PlasmaEtchingParameters) -> None: ...

class Interpolation(enum.IntEnum):
    CUSTOM: typing.ClassVar[Interpolation]  # value = <Interpolation.CUSTOM: 2>
    IDW: typing.ClassVar[Interpolation]  # value = <Interpolation.IDW: 1>
    LINEAR: typing.ClassVar[Interpolation]  # value = <Interpolation.LINEAR: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class IonBeamEtching(ProcessModel):
    def __init__(
        self,
        parameters: viennaps._core.IBEParameters,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...

class IsotropicProcess(ProcessModel):
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat = 1.0,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat,
        maskMaterial: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Mapping[
            viennaps._core.Material, typing.SupportsFloat
        ],
        defaultRate: typing.SupportsFloat = 0.0,
    ) -> None: ...

class MakeFin:
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        finWidth: typing.SupportsFloat,
        finHeight: typing.SupportsFloat,
        finTaperAngle: typing.SupportsFloat = 0.0,
        maskHeight: typing.SupportsFloat = 0,
        maskTaperAngle: typing.SupportsFloat = 0,
        halfFin: bool = False,
        material: viennaps._core.Material = ...,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        finWidth: typing.SupportsFloat,
        finHeight: typing.SupportsFloat,
        taperAngle: typing.SupportsFloat = 0.0,
        baseHeight: typing.SupportsFloat = 0.0,
        periodicBoundary: bool = False,
        makeMask: bool = False,
        material: viennaps._core.Material = ...,
    ) -> None: ...
    def apply(self) -> None:
        """
        Create a fin geometry.
        """

class MakeHole:
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        holeRadius: typing.SupportsFloat,
        holeDepth: typing.SupportsFloat,
        holeTaperAngle: typing.SupportsFloat = 0.0,
        maskHeight: typing.SupportsFloat = 0.0,
        maskTaperAngle: typing.SupportsFloat = 0.0,
        holeShape: viennaps._core.HoleShape = ...,
        material: viennaps._core.Material = ...,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        holeRadius: typing.SupportsFloat,
        holeDepth: typing.SupportsFloat,
        taperingAngle: typing.SupportsFloat = 0.0,
        baseHeight: typing.SupportsFloat = 0.0,
        periodicBoundary: bool = False,
        makeMask: bool = False,
        material: viennaps._core.Material = ...,
        holeShape: viennaps._core.HoleShape = ...,
    ) -> None: ...
    def apply(self) -> None:
        """
        Create a hole geometry.
        """

class MakePlane:
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        height: typing.SupportsFloat = 0.0,
        material: viennaps._core.Material = ...,
        addToExisting: bool = False,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        height: typing.SupportsFloat = 0.0,
        periodicBoundary: bool = False,
        material: viennaps._core.Material = ...,
    ) -> None: ...
    def apply(self) -> None:
        """
        Create a plane geometry or add plane to existing geometry.
        """

class MakeStack:
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        numLayers: typing.SupportsInt,
        layerHeight: typing.SupportsFloat,
        substrateHeight: typing.SupportsFloat = 0,
        holeRadius: typing.SupportsFloat = 0,
        trenchWidth: typing.SupportsFloat = 0,
        maskHeight: typing.SupportsFloat = 0,
        taperAngle: typing.SupportsFloat = 0,
        halfStack: bool = False,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        numLayers: typing.SupportsInt,
        layerHeight: typing.SupportsFloat,
        substrateHeight: typing.SupportsFloat,
        holeRadius: typing.SupportsFloat,
        trenchWidth: typing.SupportsFloat,
        maskHeight: typing.SupportsFloat,
        periodicBoundary: bool = False,
    ) -> None: ...
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
    class MaterialLayer:
        @typing.overload
        def __init__(self) -> None: ...
        @typing.overload
        def __init__(
            self,
            height: typing.SupportsFloat,
            width: typing.SupportsFloat,
            taperAngle: typing.SupportsFloat,
            material: viennaps._core.Material,
            isMask: bool,
        ) -> None: ...
        @property
        def height(self) -> float:
            """
            Layer thickness
            """

        @height.setter
        def height(self, arg0: typing.SupportsFloat) -> None: ...
        @property
        def isMask(self) -> bool:
            """
            true: apply cutout (mask behavior), false: no cutout
            """

        @isMask.setter
        def isMask(self, arg0: bool) -> None: ...
        @property
        def material(self) -> viennaps._core.Material:
            """
            Material type for this layer
            """

        @material.setter
        def material(self, arg0: viennaps._core.Material) -> None: ...
        @property
        def taperAngle(self) -> float:
            """
            Taper angle for cutout (degrees)
            """

        @taperAngle.setter
        def taperAngle(self, arg0: typing.SupportsFloat) -> None: ...
        @property
        def width(self) -> float:
            """
            Width of cutout for this layer
            """

        @width.setter
        def width(self, arg0: typing.SupportsFloat) -> None: ...

    @typing.overload
    def __init__(
        self,
        domain: Domain,
        trenchWidth: typing.SupportsFloat,
        trenchDepth: typing.SupportsFloat,
        trenchTaperAngle: typing.SupportsFloat = 0,
        maskHeight: typing.SupportsFloat = 0,
        maskTaperAngle: typing.SupportsFloat = 0,
        halfTrench: bool = False,
        material: viennaps._core.Material = ...,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        gridDelta: typing.SupportsFloat,
        xExtent: typing.SupportsFloat,
        yExtent: typing.SupportsFloat,
        trenchWidth: typing.SupportsFloat,
        trenchDepth: typing.SupportsFloat,
        taperingAngle: typing.SupportsFloat = 0.0,
        baseHeight: typing.SupportsFloat = 0.0,
        periodicBoundary: bool = False,
        makeMask: bool = False,
        material: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        materialLayers: collections.abc.Sequence[MakeTrench.MaterialLayer],
        halfTrench: bool = False,
    ) -> None: ...
    def apply(self) -> None:
        """
        Create a trench geometry.
        """

class MultiParticleProcess(ProcessModel):
    def __init__(self) -> None: ...
    def addIonParticle(
        self,
        sourcePower: typing.SupportsFloat,
        thetaRMin: typing.SupportsFloat = 0.0,
        thetaRMax: typing.SupportsFloat = 90.0,
        minAngle: typing.SupportsFloat = 0.0,
        B_sp: typing.SupportsFloat = -1.0,
        meanEnergy: typing.SupportsFloat = 0.0,
        sigmaEnergy: typing.SupportsFloat = 0.0,
        thresholdEnergy: typing.SupportsFloat = 0.0,
        inflectAngle: typing.SupportsFloat = 0.0,
        n: typing.SupportsFloat = 1,
        label: str = "ionFlux",
    ) -> None: ...
    @typing.overload
    def addNeutralParticle(
        self, stickingProbability: typing.SupportsFloat, label: str = "neutralFlux"
    ) -> None: ...
    @typing.overload
    def addNeutralParticle(
        self,
        materialSticking: collections.abc.Mapping[
            viennaps._core.Material, typing.SupportsFloat
        ],
        defaultStickingProbability: typing.SupportsFloat = 1.0,
        label: str = "neutralFlux",
    ) -> None: ...
    def setRateFunction(
        self,
        arg0: collections.abc.Callable[
            [collections.abc.Sequence[typing.SupportsFloat], viennaps._core.Material],
            float,
        ],
    ) -> None: ...

class OxideRegrowth(ProcessModel):
    def __init__(
        self,
        nitrideEtchRate: typing.SupportsFloat,
        oxideEtchRate: typing.SupportsFloat,
        redepositionRate: typing.SupportsFloat,
        redepositionThreshold: typing.SupportsFloat,
        redepositionTimeInt: typing.SupportsFloat,
        diffusionCoefficient: typing.SupportsFloat,
        sinkStrength: typing.SupportsFloat,
        scallopVelocity: typing.SupportsFloat,
        centerVelocity: typing.SupportsFloat,
        topHeight: typing.SupportsFloat,
        centerWidth: typing.SupportsFloat,
        stabilityFactor: typing.SupportsFloat,
    ) -> None: ...

class Planarize:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self, geometry: Domain, cutoffHeight: typing.SupportsFloat = 0.0
    ) -> None: ...
    def apply(self) -> None:
        """
        Apply the planarization.
        """

    def setCutoffPosition(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the cutoff height for the planarization.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain in the planarization.
        """

class Process:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain) -> None: ...
    @typing.overload
    def __init__(
        self,
        domain: Domain,
        model: ProcessModelBase,
        duration: typing.SupportsFloat = 0.0,
    ) -> None: ...
    def apply(self) -> None:
        """
        Run the process.
        """

    def calculateFlux(self) -> viennals._core.Mesh:
        """
        Perform a single-pass flux calculation.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the process domain.
        """

    def setFluxEngineType(self, arg0: viennaps._core.FluxEngineType) -> None:
        """
        Set the flux engine type (CPU or GPU).
        """

    @typing.overload
    def setParameters(self, parameters: viennaps._core.AdvectionParameters) -> None:
        """
        Set the advection parameters for the process.
        """

    @typing.overload
    def setParameters(self, parameters: viennaps._core.RayTracingParameters) -> None:
        """
        Set the ray tracing parameters for the process.
        """

    @typing.overload
    def setParameters(self, parameters: viennaps._core.CoverageParameters) -> None:
        """
        Set the coverage parameters for the process.
        """

    @typing.overload
    def setParameters(
        self, parameters: viennaps._core.AtomicLayerProcessParameters
    ) -> None:
        """
        Set the atomic layer parameters for the process.
        """

    def setProcessDuration(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the process duration.
        """

    def setProcessModel(self, arg0: ProcessModelBase) -> None:
        """
        Set the process model. This has to be a pre-configured process model.
        """

class ProcessModel(ProcessModelBase):
    @staticmethod
    def setAdvectionCallback(*args, **kwargs) -> None: ...
    @staticmethod
    def setGeometricModel(*args, **kwargs) -> None: ...
    @staticmethod
    def setVelocityField(*args, **kwargs) -> None: ...
    def __init__(self) -> None: ...
    def getAdvectionCallback(self) -> ...: ...
    def getGeometricModel(self) -> ...: ...
    def getPrimaryDirection(
        self,
    ) -> typing.Annotated[list[float], "FixedSize(3)"] | None: ...
    def getProcessName(self) -> str | None: ...
    def getSurfaceModel(self) -> ...: ...
    def getVelocityField(self) -> ...: ...
    def setPrimaryDirection(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    def setProcessName(self, arg0: str) -> None: ...
    def setSurfaceModel(self, arg0: ...) -> None: ...

class ProcessModelBase:
    pass

class RateGrid:
    def __init__(self) -> None: ...
    def interpolate(
        self,
        coord: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> float: ...
    def loadFromCSV(self, filename: str) -> bool: ...
    def setCustomInterpolator(self, function: collections.abc.Callable) -> None: ...
    def setIDWNeighbors(self, k: typing.SupportsInt) -> None: ...
    @typing.overload
    def setInterpolationMode(self, mode: Interpolation) -> None: ...
    @typing.overload
    def setInterpolationMode(self, mode: str) -> None: ...
    def setOffset(
        self,
        offset: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None: ...

class Reader:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, fileName: str) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain, fileName: str) -> None: ...
    def apply(self) -> None:
        """
        Read the domain from the specified file.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain to read into.
        """

    def setFileName(self, arg0: str) -> None:
        """
        Set the input file name to read (should end with .vpsd).
        """

class SF6C4F8Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        ionFlux: typing.SupportsFloat,
        etchantFlux: typing.SupportsFloat,
        meanEnergy: typing.SupportsFloat,
        sigmaEnergy: typing.SupportsFloat,
        ionExponent: typing.SupportsFloat = 300.0,
        etchStopDepth: typing.SupportsFloat = -1.7976931348623157e308,
    ) -> None: ...
    @typing.overload
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> viennaps._core.PlasmaEtchingParameters: ...
    def setParameters(self, arg0: viennaps._core.PlasmaEtchingParameters) -> None: ...

class SF6O2Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        ionFlux: typing.SupportsFloat,
        etchantFlux: typing.SupportsFloat,
        oxygenFlux: typing.SupportsFloat,
        meanIonEnergy: typing.SupportsFloat = 100.0,
        sigmaIonEnergy: typing.SupportsFloat = 10.0,
        ionExponent: typing.SupportsFloat = 100.0,
        oxySputterYield: typing.SupportsFloat = 3.0,
        etchStopDepth: typing.SupportsFloat = -1.7976931348623157e308,
    ) -> None: ...
    @typing.overload
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> viennaps._core.PlasmaEtchingParameters: ...
    def setParameters(self, arg0: viennaps._core.PlasmaEtchingParameters) -> None: ...

class SelectiveEpitaxy(ProcessModel):
    def __init__(
        self,
        materialRates: collections.abc.Sequence[
            tuple[viennaps._core.Material, typing.SupportsFloat]
        ],
        rate111: typing.SupportsFloat = 0.5,
        rate100: typing.SupportsFloat = 1.0,
    ) -> None: ...

class SingleParticleALD(ProcessModel):
    def __init__(
        self,
        stickingProbability: typing.SupportsFloat,
        numCycles: typing.SupportsInt,
        growthPerCycle: typing.SupportsFloat,
        totalCycles: typing.SupportsInt,
        coverageTimeStep: typing.SupportsFloat,
        evFlux: typing.SupportsFloat,
        inFlux: typing.SupportsFloat,
        s0: typing.SupportsFloat,
        gasMFP: typing.SupportsFloat,
    ) -> None: ...

class SingleParticleProcess(ProcessModel):
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat = 1.0,
        stickingProbability: typing.SupportsFloat = 1.0,
        sourceExponent: typing.SupportsFloat = 1.0,
        maskMaterial: viennaps._core.Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat,
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Mapping[
            viennaps._core.Material, typing.SupportsFloat
        ],
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
    ) -> None: ...

class SphereDistribution(ProcessModel):
    @typing.overload
    def __init__(
        self, radius: typing.SupportsFloat, mask: viennals.d3.Domain
    ) -> None: ...
    @typing.overload
    def __init__(self, radius: typing.SupportsFloat) -> None: ...

class StencilLocalLaxFriedrichsScalar:
    @staticmethod
    def setMaxDissipation(maxDissipation: typing.SupportsFloat) -> None: ...

class TEOSDeposition(ProcessModel):
    def __init__(
        self,
        stickingProbabilityP1: typing.SupportsFloat,
        rateP1: typing.SupportsFloat,
        orderP1: typing.SupportsFloat,
        stickingProbabilityP2: typing.SupportsFloat = 0.0,
        rateP2: typing.SupportsFloat = 0.0,
        orderP2: typing.SupportsFloat = 0.0,
    ) -> None: ...

class TEOSPECVD(ProcessModel):
    def __init__(
        self,
        stickingProbabilityRadical: typing.SupportsFloat,
        depositionRateRadical: typing.SupportsFloat,
        depositionRateIon: typing.SupportsFloat,
        exponentIon: typing.SupportsFloat,
        stickingProbabilityIon: typing.SupportsFloat = 1.0,
        reactionOrderRadical: typing.SupportsFloat = 1.0,
        reactionOrderIon: typing.SupportsFloat = 1.0,
        minAngleIon: typing.SupportsFloat = 0.0,
    ) -> None: ...

class ToDiskMesh:
    @typing.overload
    def __init__(self, domain: Domain, mesh: viennals._core.Mesh) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain in the mesh converter.
        """

    def setMesh(self, arg0: viennals._core.Mesh) -> None:
        """
        Set the mesh in the mesh converter
        """

class WetEtching(ProcessModel):
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Sequence[
            tuple[viennaps._core.Material, typing.SupportsFloat]
        ],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        direction100: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        direction010: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        rate100: typing.SupportsFloat,
        rate110: typing.SupportsFloat,
        rate111: typing.SupportsFloat,
        rate311: typing.SupportsFloat,
        materialRates: collections.abc.Sequence[
            tuple[viennaps._core.Material, typing.SupportsFloat]
        ],
    ) -> None: ...

class Writer:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain, fileName: str) -> None: ...
    def apply(self) -> None:
        """
        Write the domain to the specified file.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain to be written to a file.
        """

    def setFileName(self, arg0: str) -> None:
        """
        Set the output file name (should end with .vpsd).
        """
