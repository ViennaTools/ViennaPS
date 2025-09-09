"""
ViennaPS is a header-only C++ process simulation library which includes surface and volume representations, a ray tracer, and physical models for the simulation of microelectronic fabrication processes. The main design goals are simplicity and efficiency, tailored towards scientific simulations.
"""

from __future__ import annotations
import collections.abc
import enum
import typing
import viennals3d.viennals3d
from . import constants
from . import gpu
from . import ray
from . import util

__all__: list[str] = [
    "AdvectionCallback",
    "AdvectionParameters",
    "AtomicLayerProcess",
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
    "ray",
    "setNumThreads",
    "util",
    "version",
]

class AdvectionCallback:
    domain: Domain
    def __init__(self) -> None: ...
    def applyPostAdvect(self, arg0: typing.SupportsFloat) -> bool: ...
    def applyPreAdvect(self, arg0: typing.SupportsFloat) -> bool: ...

class AdvectionParameters:
    checkDissipation: bool
    ignoreVoids: bool
    integrationScheme: viennals3d.viennals3d.IntegrationSchemeEnum
    velocityOutput: bool
    def __init__(self) -> None: ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the advection parameters to a metadata dict.
        """

    def toMetaDataString(self) -> str:
        """
        Convert the advection parameters to a metadata string.
        """

    @property
    def dissipationAlpha(self) -> float: ...
    @dissipationAlpha.setter
    def dissipationAlpha(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def timeStepRatio(self) -> float: ...
    @timeStepRatio.setter
    def timeStepRatio(self, arg0: typing.SupportsFloat) -> None: ...

class AtomicLayerProcess:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain, model: ProcessModel) -> None: ...
    def apply(self) -> None:
        """
        Run the process.
        """

    def disableRandomSeeds(self) -> None:
        """
        Disable random seeds for the ray tracer. This will make the process results deterministic.
        """

    def enableRandomSeeds(self) -> None:
        """
        Enable random seeds for the ray tracer. This will make the process results non-deterministic.
        """

    def setCoverageTimeStep(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the time step for the coverage calculation.
        """

    def setDesorptionRates(
        self, arg0: collections.abc.Sequence[typing.SupportsFloat]
    ) -> None:
        """
        Set the desorption rate for each surface point.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the process domain.
        """

    def setIntegrationScheme(
        self, arg0: viennals3d.viennals3d.IntegrationSchemeEnum
    ) -> None:
        """
        Set the integration scheme for solving the level-set equation. Possible integration schemes are specified in lsIntegrationSchemeEnum.
        """

    def setNumCycles(self, arg0: typing.SupportsInt) -> None:
        """
        Set the number of cycles for the process.
        """

    def setNumberOfRaysPerPoint(self, arg0: typing.SupportsInt) -> None:
        """
        Set the number of rays to traced for each particle in the process. The number is per point in the process geometry.
        """

    def setProcessModel(self, arg0: ProcessModel) -> None:
        """
        Set the process model. This has to be a pre-configured process model.
        """

    def setPulseTime(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the pulse time.
        """

    def setSourceDirection(self, arg0: ...) -> None:
        """
        Set source direction of the process.
        """

class BoxDistribution(ProcessModel):
    @typing.overload
    def __init__(
        self,
        halfAxes: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        gridDelta: typing.SupportsFloat,
        mask: viennals3d.viennals3d.Domain,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        halfAxes: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        gridDelta: typing.SupportsFloat,
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
    def __init__(self, parameters: CF4O2Parameters) -> None: ...
    def getParameters(self) -> CF4O2Parameters: ...
    def setParameters(self, arg0: CF4O2Parameters) -> None: ...

class CF4O2Parameters:
    Ions: CF4O2ParametersIons
    Mask: CF4O2ParametersMask
    Passivation: CF4O2ParametersPassivation
    Si: CF4O2ParametersSi
    SiGe: CF4O2ParametersSiGe
    fluxIncludeSticking: bool
    def __init__(self) -> None: ...
    @property
    def etchStopDepth(self) -> float: ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def etchantFlux(self) -> float: ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def gamma_C(self) -> dict[Material, float]: ...
    @gamma_C.setter
    def gamma_C(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def gamma_C_oxidized(self) -> dict[Material, float]: ...
    @gamma_C_oxidized.setter
    def gamma_C_oxidized(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def gamma_F(self) -> dict[Material, float]: ...
    @gamma_F.setter
    def gamma_F(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def gamma_F_oxidized(self) -> dict[Material, float]: ...
    @gamma_F_oxidized.setter
    def gamma_F_oxidized(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def gamma_O(self) -> dict[Material, float]: ...
    @gamma_O.setter
    def gamma_O(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def gamma_O_passivated(self) -> dict[Material, float]: ...
    @gamma_O_passivated.setter
    def gamma_O_passivated(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def ionFlux(self) -> float: ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def oxygenFlux(self) -> float: ...
    @oxygenFlux.setter
    def oxygenFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def polymerFlux(self) -> float: ...
    @polymerFlux.setter
    def polymerFlux(self, arg0: typing.SupportsFloat) -> None: ...

class CF4O2ParametersIons:
    def __init__(self) -> None: ...
    @property
    def exponent(self) -> float: ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def inflectAngle(self) -> float: ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def meanEnergy(self) -> float: ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def minAngle(self) -> float: ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def n_l(self) -> float: ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def sigmaEnergy(self) -> float: ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat) -> None: ...

class CF4O2ParametersMask:
    def __init__(self) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class CF4O2ParametersPassivation:
    def __init__(self) -> None: ...
    @property
    def A_C_ie(self) -> float: ...
    @A_C_ie.setter
    def A_C_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_O_ie(self) -> float: ...
    @A_O_ie.setter
    def A_O_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_C_ie(self) -> float: ...
    @Eth_C_ie.setter
    def Eth_C_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_O_ie(self) -> float: ...
    @Eth_O_ie.setter
    def Eth_O_ie(self, arg0: typing.SupportsFloat) -> None: ...

class CF4O2ParametersSi:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_sigma(self) -> float: ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def k_sigma(self) -> float: ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class CF4O2ParametersSiGe:
    def __init__(self) -> None: ...
    def k_sigma_SiGe(self, arg0: typing.SupportsFloat) -> float: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_sigma(self) -> float: ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def k_sigma(self) -> float: ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat) -> None: ...

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
        maskMaterials: collections.abc.Sequence[Material] = ...,
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
        levelSets: collections.abc.Sequence[viennals3d.viennals3d.Domain],
        materialMap: viennals3d.viennals3d.MaterialMap = None,
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

    def getCellGrid(self) -> viennals3d.viennals3d.Mesh:
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

    def getSurface(self) -> viennals3d.viennals3d.Domain:
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
        maskMaterial: Material = ...,
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
        maskMaterial: collections.abc.Sequence[Material],
        calculateVisibility: bool = True,
    ) -> None: ...
    @typing.overload
    def __init__(self, rateSets: collections.abc.Sequence[RateSet]) -> None: ...
    @typing.overload
    def __init__(self, rateSet: RateSet) -> None: ...

class Domain:
    @staticmethod
    def disableMetaData() -> None:
        """
        Disable adding meta data to domain.
        """

    @staticmethod
    def enableMetaData(level: MetaDataLevel = ...) -> None:
        """
        Enable adding meta data from processes to domain.
        """

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
        boundary: viennals3d.viennals3d.BoundaryConditionEnum = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        bounds: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(6)"
        ],
        boundaryConditions: typing.Annotated[
            collections.abc.Sequence[viennals3d.viennals3d.BoundaryConditionEnum],
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
        self, arg0: viennals3d.viennals3d.Domain, arg1: ...
    ) -> None: ...
    def clear(self) -> None: ...
    def clearMetaData(self, clearDomainData: bool = False) -> None:
        """
        Clear meta data from domain.
        """

    def deepCopy(self, arg0: Domain) -> None: ...
    def duplicateTopLevelSet(self, arg0: Material) -> None:
        """
        Duplicate the top level set. Should be used before a deposition process.
        """

    def generateCellSet(
        self, arg0: typing.SupportsFloat, arg1: Material, arg2: bool
    ) -> None:
        """
        Generate the cell set.
        """

    def getBoundaryConditions(
        self,
    ) -> typing.Annotated[
        list[viennals3d.viennals3d.BoundaryConditionEnum], "FixedSize(3)"
    ]:
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

    def getCellSet(self) -> ...:
        """
        Get the cell set.
        """

    def getGrid(self) -> viennals3d.viennals3d.hrleGrid:
        """
        Get the grid
        """

    def getGridDelta(self) -> float:
        """
        Get the grid delta.
        """

    def getLevelSets(self) -> list[viennals3d.viennals3d.Domain]: ...
    def getMaterialMap(self) -> MaterialMap: ...
    def getMetaData(self) -> dict[str, list[float]]:
        """
        Get meta data (e.g. process data) stored in the domain
        """

    def getSetup(self) -> DomainSetup:
        """
        Get the domain setup.
        """

    def insertNextLevelSet(
        self, levelset: viennals3d.viennals3d.Domain, wrapLowerLevelSet: bool = True
    ) -> None:
        """
        Insert a level set to domain.
        """

    def insertNextLevelSetAsMaterial(
        self,
        levelSet: viennals3d.viennals3d.Domain,
        material: Material,
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
    def removeMaterial(self, arg0: Material) -> None: ...
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
    def saveSurfaceMesh(self, filename: str, addMaterialIds: bool = False) -> None:
        """
        Save the surface of the domain.
        """

    def saveVolumeMesh(
        self, filename: str, wrappingLayerEpsilon: typing.SupportsFloat = 0.01
    ) -> None:
        """
        Save the volume representation of the domain.
        """

    def setMaterialMap(self, arg0: MaterialMap) -> None: ...
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
        boundary: viennals3d.viennals3d.BoundaryConditionEnum = ...,
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
        boundary: viennals3d.viennals3d.BoundaryConditionEnum = ...,
    ) -> None: ...
    def boundaryCons(
        self,
    ) -> typing.Annotated[
        list[viennals3d.viennals3d.BoundaryConditionEnum], "FixedSize(3)"
    ]: ...
    def bounds(self) -> typing.Annotated[list[float], "FixedSize(6)"]: ...
    def check(self) -> None: ...
    def grid(self) -> viennals3d.viennals3d.hrleGrid: ...
    def gridDelta(self) -> float: ...
    def halveXAxis(self) -> None: ...
    def halveYAxis(self) -> None: ...
    def hasPeriodicBoundary(self) -> bool: ...
    def isValid(self) -> bool: ...
    def print(self) -> None: ...
    def xExtent(self) -> float: ...
    def yExtent(self) -> float: ...

class FaradayCageEtching(ProcessModel):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, maskMaterials: collections.abc.Sequence[Material]) -> None: ...
    @typing.overload
    def __init__(
        self,
        maskMaterials: collections.abc.Sequence[Material],
        parameters: FaradayCageParameters,
    ) -> None: ...
    def getParameters(self) -> FaradayCageParameters: ...
    def setParameters(self, arg0: FaradayCageParameters) -> None: ...

class FaradayCageParameters:
    ibeParams: IBEParameters
    def __init__(self) -> None: ...
    @property
    def cageAngle(self) -> float: ...
    @cageAngle.setter
    def cageAngle(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonEtching(ProcessModel):
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        ionFlux: typing.SupportsFloat,
        etchantFlux: typing.SupportsFloat,
        polyFlux: typing.SupportsFloat,
        meanIonEnergy: typing.SupportsFloat = 100.0,
        sigmaIonEnergy: typing.SupportsFloat = 10.0,
        ionExponent: typing.SupportsFloat = 100.0,
        deltaP: typing.SupportsFloat = 0.0,
        etchStopDepth: typing.SupportsFloat = -1.7976931348623157e308,
    ) -> None: ...
    @typing.overload
    def __init__(self, parameters: FluorocarbonParameters) -> None: ...
    def getParameters(self) -> FluorocarbonParameters: ...
    def setParameters(self, arg0: FluorocarbonParameters) -> None: ...

class FluorocarbonParameters:
    Ions: FluorocarbonParametersIons
    Mask: FluorocarbonParametersMask
    Polymer: FluorocarbonParametersPolymer
    Si: FluorocarbonParametersSi
    Si3N4: FluorocarbonParametersSi3N4
    SiO2: FluorocarbonParametersSiO2
    def __init__(self) -> None: ...
    @property
    def delta_p(self) -> float: ...
    @delta_p.setter
    def delta_p(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def etchStopDepth(self) -> float: ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def etchantFlux(self) -> float: ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def ionFlux(self) -> float: ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def polyFlux(self) -> float: ...
    @polyFlux.setter
    def polyFlux(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersIons:
    def __init__(self) -> None: ...
    @property
    def exponent(self) -> float: ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def inflectAngle(self) -> float: ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def meanEnergy(self) -> float: ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def minAngle(self) -> float: ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def n_l(self) -> float: ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def sigmaEnergy(self) -> float: ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersMask:
    def __init__(self) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_e(self) -> float: ...
    @beta_e.setter
    def beta_e(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_p(self) -> float: ...
    @beta_p.setter
    def beta_p(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersPolymer:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersSi:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def E_a(self) -> float: ...
    @E_a.setter
    def E_a(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def K(self) -> float: ...
    @K.setter
    def K(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersSi3N4:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def E_a(self) -> float: ...
    @E_a.setter
    def E_a(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def K(self) -> float: ...
    @K.setter
    def K(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParametersSiO2:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def E_a(self) -> float: ...
    @E_a.setter
    def E_a(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def K(self) -> float: ...
    @K.setter
    def K(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

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
            collections.abc.Sequence[viennals3d.viennals3d.BoundaryConditionEnum],
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
    ) -> viennals3d.viennals3d.Domain: ...
    def print(self) -> None:
        """
        Print the geometry contents.
        """

    def setBoundaryConditions(
        self,
        arg0: collections.abc.Sequence[viennals3d.viennals3d.BoundaryConditionEnum],
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
    ) -> viennals3d.viennals3d.Domain: ...
    def makeCylinderStencil(
        self,
        position: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
        radius: typing.SupportsFloat,
        height: typing.SupportsFloat,
        angle: typing.SupportsFloat = 0.0,
    ) -> viennals3d.viennals3d.Domain: ...
    def makeMask(
        self, base: typing.SupportsFloat, height: typing.SupportsFloat
    ) -> viennals3d.viennals3d.Domain: ...
    def makeSubstrate(
        self, base: typing.SupportsFloat
    ) -> viennals3d.viennals3d.Domain: ...

class HBrO2Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> PlasmaEtchingParameters: ...
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
    def __init__(self, parameters: PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> PlasmaEtchingParameters: ...
    def setParameters(self, arg0: PlasmaEtchingParameters) -> None: ...

class HoleShape(enum.IntEnum):
    FULL: typing.ClassVar[HoleShape]  # value = <HoleShape.FULL: 0>
    HALF: typing.ClassVar[HoleShape]  # value = <HoleShape.HALF: 1>
    QUARTER: typing.ClassVar[HoleShape]  # value = <HoleShape.QUARTER: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class IBEParameters:
    yieldFunction: collections.abc.Callable[[typing.SupportsFloat], float]
    def __init__(self) -> None: ...
    @property
    def exponent(self) -> float: ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def inflectAngle(self) -> float: ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def materialPlaneWaferRate(self) -> dict[Material, float]: ...
    @materialPlaneWaferRate.setter
    def materialPlaneWaferRate(
        self, arg0: collections.abc.Mapping[Material, typing.SupportsFloat]
    ) -> None: ...
    @property
    def meanEnergy(self) -> float: ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def minAngle(self) -> float: ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def n_l(self) -> float: ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def planeWaferRate(self) -> float: ...
    @planeWaferRate.setter
    def planeWaferRate(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def redepositionRate(self) -> float: ...
    @redepositionRate.setter
    def redepositionRate(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def redepositionThreshold(self) -> float: ...
    @redepositionThreshold.setter
    def redepositionThreshold(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def sigmaEnergy(self) -> float: ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def thresholdEnergy(self) -> float: ...
    @thresholdEnergy.setter
    def thresholdEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def tiltAngle(self) -> float: ...
    @tiltAngle.setter
    def tiltAngle(self, arg0: typing.SupportsFloat) -> None: ...

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
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, maskMaterials: collections.abc.Sequence[Material]) -> None: ...
    @typing.overload
    def __init__(
        self,
        maskMaterials: collections.abc.Sequence[Material],
        parameters: IBEParameters,
    ) -> None: ...
    def getParameters(self) -> IBEParameters: ...
    def setParameters(self, arg0: IBEParameters) -> None: ...

class IsotropicProcess(ProcessModel):
    @typing.overload
    def __init__(
        self, rate: typing.SupportsFloat = 1.0, maskMaterial: Material = ...
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat,
        maskMaterial: collections.abc.Sequence[Material],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Mapping[Material, typing.SupportsFloat],
        defaultRate: typing.SupportsFloat = 0.0,
    ) -> None: ...

class Length:
    @staticmethod
    def convertAngstrom() -> float: ...
    @staticmethod
    def convertCentimeter() -> float: ...
    @staticmethod
    def convertMeter() -> float: ...
    @staticmethod
    def convertMicrometer() -> float: ...
    @staticmethod
    def convertMillimeter() -> float: ...
    @staticmethod
    def convertNanometer() -> float: ...
    @staticmethod
    def getInstance() -> Length: ...
    @staticmethod
    def setUnit(arg0: str) -> None: ...
    @staticmethod
    def toShortString() -> str: ...
    @staticmethod
    def toString() -> str: ...

class LengthUnit(enum.IntEnum):
    ANGSTROM: typing.ClassVar[LengthUnit]  # value = <LengthUnit.ANGSTROM: 5>
    CENTIMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.CENTIMETER: 1>
    METER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.METER: 0>
    MICROMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.MICROMETER: 3>
    MILLIMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.MILLIMETER: 2>
    NANOMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.NANOMETER: 4>
    UNDEFINED: typing.ClassVar[LengthUnit]  # value = <LengthUnit.UNDEFINED: 6>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class LogLevel:
    """
    Members:

      ERROR

      WARNING

      INFO

      INTERMEDIATE

      TIMING

      DEBUG
    """

    DEBUG: typing.ClassVar[LogLevel]  # value = <LogLevel.DEBUG: 5>
    ERROR: typing.ClassVar[LogLevel]  # value = <LogLevel.ERROR: 0>
    INFO: typing.ClassVar[LogLevel]  # value = <LogLevel.INFO: 2>
    INTERMEDIATE: typing.ClassVar[LogLevel]  # value = <LogLevel.INTERMEDIATE: 3>
    TIMING: typing.ClassVar[LogLevel]  # value = <LogLevel.TIMING: 4>
    WARNING: typing.ClassVar[LogLevel]  # value = <LogLevel.WARNING: 1>
    __members__: typing.ClassVar[
        dict[str, LogLevel]
    ]  # value = {'ERROR': <LogLevel.ERROR: 0>, 'WARNING': <LogLevel.WARNING: 1>, 'INFO': <LogLevel.INFO: 2>, 'INTERMEDIATE': <LogLevel.INTERMEDIATE: 3>, 'TIMING': <LogLevel.TIMING: 4>, 'DEBUG': <LogLevel.DEBUG: 5>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Logger:
    @staticmethod
    def appendToLogFile(arg0: str) -> bool: ...
    @staticmethod
    def closeLogFile() -> None: ...
    @staticmethod
    def getInstance() -> Logger: ...
    @staticmethod
    def getLogLevel() -> int: ...
    @staticmethod
    def setLogFile(arg0: str) -> bool: ...
    @staticmethod
    def setLogLevel(arg0: LogLevel) -> None: ...
    def addDebug(self, arg0: str) -> Logger: ...
    def addError(self, s: str, shouldAbort: bool = True) -> Logger: ...
    def addInfo(self, arg0: str) -> Logger: ...
    @typing.overload
    def addTiming(self, arg0: str, arg1: typing.SupportsFloat) -> Logger: ...
    @typing.overload
    def addTiming(
        self, arg0: str, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat
    ) -> Logger: ...
    def addWarning(self, arg0: str) -> Logger: ...
    def print(self) -> None: ...

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
        material: Material = ...,
        maskMaterial: Material = ...,
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
        material: Material = ...,
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
        holeShape: HoleShape = ...,
        material: Material = ...,
        maskMaterial: Material = ...,
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
        material: Material = ...,
        holeShape: HoleShape = ...,
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
        material: Material = ...,
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
        material: Material = ...,
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
        maskMaterial: Material = ...,
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
        material: Material = ...,
        maskMaterial: Material = ...,
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
        material: Material = ...,
    ) -> None: ...
    def apply(self) -> None:
        """
        Create a trench geometry.
        """

class Material(enum.IntEnum):
    """
    Material types for domain and level sets.
    """

    Air: typing.ClassVar[Material]  # value = <Material.Air: 18>
    Al2O3: typing.ClassVar[Material]  # value = <Material.Al2O3: 11>
    Cu: typing.ClassVar[Material]  # value = <Material.Cu: 14>
    Dielectric: typing.ClassVar[Material]  # value = <Material.Dielectric: 16>
    GAS: typing.ClassVar[Material]  # value = <Material.GAS: 19>
    GaN: typing.ClassVar[Material]  # value = <Material.GaN: 9>
    HfO2: typing.ClassVar[Material]  # value = <Material.HfO2: 12>
    Mask: typing.ClassVar[Material]  # value = <Material.Mask: 0>
    Metal: typing.ClassVar[Material]  # value = <Material.Metal: 17>
    PolySi: typing.ClassVar[Material]  # value = <Material.PolySi: 8>
    Polymer: typing.ClassVar[Material]  # value = <Material.Polymer: 15>
    Si: typing.ClassVar[Material]  # value = <Material.Si: 1>
    Si3N4: typing.ClassVar[Material]  # value = <Material.Si3N4: 3>
    SiC: typing.ClassVar[Material]  # value = <Material.SiC: 6>
    SiGe: typing.ClassVar[Material]  # value = <Material.SiGe: 7>
    SiN: typing.ClassVar[Material]  # value = <Material.SiN: 4>
    SiO2: typing.ClassVar[Material]  # value = <Material.SiO2: 2>
    SiON: typing.ClassVar[Material]  # value = <Material.SiON: 5>
    TiN: typing.ClassVar[Material]  # value = <Material.TiN: 13>
    Undefined: typing.ClassVar[Material]  # value = <Material.Undefined: -1>
    W: typing.ClassVar[Material]  # value = <Material.W: 10>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class MaterialMap:
    @staticmethod
    def getMaterialName(arg0: Material) -> str:
        """
        Get the name of a material.
        """

    @staticmethod
    def isMaterial(arg0: typing.SupportsFloat, arg1: Material) -> bool: ...
    @staticmethod
    def mapToMaterial(arg0: typing.SupportsFloat) -> Material:
        """
        Map a float to a material.
        """

    def __init__(self) -> None: ...
    def getMaterialAtIdx(self, arg0: typing.SupportsInt) -> Material: ...
    def getMaterialMap(self) -> viennals3d.viennals3d.MaterialMap: ...
    def insertNextMaterial(self, material: Material = ...) -> None: ...
    def size(self) -> int: ...

class MetaDataLevel(enum.IntEnum):
    FULL: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.FULL: 3>
    GRID: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.GRID: 1>
    NONE: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.NONE: 0>
    PROCESS: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.PROCESS: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
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
        materialSticking: collections.abc.Mapping[Material, typing.SupportsFloat],
        defaultStickingProbability: typing.SupportsFloat = 1.0,
        label: str = "neutralFlux",
    ) -> None: ...
    def setRateFunction(
        self,
        arg0: collections.abc.Callable[
            [collections.abc.Sequence[typing.SupportsFloat], Material], float
        ],
    ) -> None: ...

class NormalizationType(enum.IntEnum):
    MAX: typing.ClassVar[NormalizationType]  # value = <NormalizationType.MAX: 1>
    SOURCE: typing.ClassVar[NormalizationType]  # value = <NormalizationType.SOURCE: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

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

class PlasmaEtchingParameters:
    Ions: PlasmaEtchingParametersIons
    Mask: PlasmaEtchingParametersMask
    Passivation: PlasmaEtchingParametersPassivation
    Substrate: PlasmaEtchingParametersSubstrate
    def __init__(self) -> None: ...
    @property
    def beta_E(self) -> dict[int, float]: ...
    @beta_E.setter
    def beta_E(
        self, arg0: collections.abc.Mapping[typing.SupportsInt, typing.SupportsFloat]
    ) -> None: ...
    @property
    def beta_P(self) -> dict[int, float]: ...
    @beta_P.setter
    def beta_P(
        self, arg0: collections.abc.Mapping[typing.SupportsInt, typing.SupportsFloat]
    ) -> None: ...
    @property
    def etchStopDepth(self) -> float: ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def etchantFlux(self) -> float: ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def ionFlux(self) -> float: ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def passivationFlux(self) -> float: ...
    @passivationFlux.setter
    def passivationFlux(self, arg0: typing.SupportsFloat) -> None: ...

class PlasmaEtchingParametersIons:
    def __init__(self) -> None: ...
    @property
    def exponent(self) -> float: ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def inflectAngle(self) -> float: ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def meanEnergy(self) -> float: ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def minAngle(self) -> float: ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def n_l(self) -> float: ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def sigmaEnergy(self) -> float: ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def thetaRMax(self) -> float: ...
    @thetaRMax.setter
    def thetaRMax(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def thetaRMin(self) -> float: ...
    @thetaRMin.setter
    def thetaRMin(self, arg0: typing.SupportsFloat) -> None: ...

class PlasmaEtchingParametersMask:
    def __init__(self) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class PlasmaEtchingParametersPassivation:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...

class PlasmaEtchingParametersPolymer:
    def __init__(self) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class PlasmaEtchingParametersSubstrate:
    def __init__(self) -> None: ...
    @property
    def A_ie(self) -> float: ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def A_sp(self) -> float: ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_ie(self) -> float: ...
    @B_ie.setter
    def B_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def B_sp(self) -> float: ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_ie(self) -> float: ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def Eth_sp(self) -> float: ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_sigma(self) -> float: ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def k_sigma(self) -> float: ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rho(self) -> float: ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat) -> None: ...

class Process:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, domain: Domain) -> None: ...
    @typing.overload
    def __init__(
        self, domain: Domain, model: ProcessModel, duration: typing.SupportsFloat
    ) -> None: ...
    def apply(self) -> None:
        """
        Run the process.
        """

    def calculateFlux(self) -> viennals3d.viennals3d.Mesh:
        """
        Perform a single-pass flux calculation.
        """

    def disableAdvectionVelocityOutput(self) -> None:
        """
        Disable the output of the advection velocity field on the ls-mesh.
        """

    def disableFluxSmoothing(self) -> None:
        """
        Disable flux smoothing
        """

    def disableRandomSeeds(self) -> None:
        """
        Disable random seeds for the ray tracer. This will make the process results deterministic.
        """

    def enableAdvectionVelocityOutput(self) -> None:
        """
        Enable the output of the advection velocity field on the ls-mesh.
        """

    def enableFluxSmoothing(self) -> None:
        """
        Enable flux smoothing. The flux at each surface point, calculated by the ray tracer, is averaged over the surface point neighbors.
        """

    def enableRandomSeeds(self) -> None:
        """
        Enable random seeds for the ray tracer. This will make the process results non-deterministic.
        """

    def getAdvectionParameters(self) -> AdvectionParameters:
        """
        Get the advection parameters for the process.
        """

    def getProcessDuration(self) -> float:
        """
        Returns the duration of the recently run process. This duration can sometimes slightly vary from the set process duration, due to the maximum time step according to the CFL condition.
        """

    def getRayTracingParameters(self) -> RayTracingParameters:
        """
        Get the ray tracing parameters for the process.
        """

    def setAdvectionParameters(self, arg0: AdvectionParameters) -> None:
        """
        Set the advection parameters for the process.
        """

    def setCoverageDeltaThreshold(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the threshold for the coverage delta metric to reach convergence.
        """

    def setDomain(self, arg0: Domain) -> None:
        """
        Set the process domain.
        """

    def setIntegrationScheme(
        self, arg0: viennals3d.viennals3d.IntegrationSchemeEnum
    ) -> None:
        """
        Set the integration scheme for solving the level-set equation. Possible integration schemes are specified in viennals::IntegrationSchemeEnum.
        """

    def setMaxCoverageInitIterations(self, arg0: typing.SupportsInt) -> None:
        """
        Set the number of iterations to initialize the coverages.
        """

    def setNumberOfRaysPerPoint(self, arg0: typing.SupportsInt) -> None:
        """
        Set the number of rays to traced for each particle in the process. The number is per point in the process geometry.
        """

    def setProcessDuration(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the process duration.
        """

    def setProcessModel(self, arg0: ProcessModel) -> None:
        """
        Set the process model. This has to be a pre-configured process model.
        """

    def setRayTracingDiskRadius(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the radius of the disk used for ray tracing. This disk is used for the intersection calculations at each surface point.
        """

    def setRayTracingParameters(self, arg0: RayTracingParameters) -> None:
        """
        Set the ray tracing parameters for the process.
        """

    def setSourceDirection(self, arg0: ...) -> None:
        """
        Set source direction of the process.
        """

    def setTimeStepRatio(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the CFL condition to use during advection. The CFL condition sets the maximum distance a surface can be moved during one advection step. It MUST be below 0.5 to guarantee numerical stability. Defaults to 0.4999.
        """

class ProcessModel:
    def __init__(self) -> None: ...
    def getPrimaryDirection(
        self,
    ) -> typing.Annotated[list[float], "FixedSize(3)"] | None: ...
    def getProcessName(self) -> str | None: ...
    def setPrimaryDirection(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    def setProcessName(self, arg0: str) -> None: ...

class ProcessParams:
    def __init__(self) -> None: ...
    @typing.overload
    def getScalarData(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def getScalarData(self, arg0: typing.SupportsInt) -> float: ...
    @typing.overload
    def getScalarData(self, arg0: str) -> float: ...
    @typing.overload
    def getScalarData(self) -> list[float]: ...
    @typing.overload
    def getScalarData(self) -> list[float]: ...
    def getScalarDataIndex(self, arg0: str) -> int: ...
    def getScalarDataLabel(self, arg0: typing.SupportsInt) -> str: ...
    def insertNextScalar(self, arg0: typing.SupportsFloat, arg1: str) -> None: ...

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

class RateSet:
    calculateVisibility: bool
    def __init__(
        self,
        direction: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ] = [0.0, 0.0, 0.0],
        directionalVelocity: typing.SupportsFloat = 0.0,
        isotropicVelocity: typing.SupportsFloat = 0.0,
        maskMaterials: collections.abc.Sequence[Material] = ...,
        calculateVisibility: bool = True,
    ) -> None: ...
    def print(self) -> None: ...
    @property
    def direction(self) -> typing.Annotated[list[float], "FixedSize(3)"]: ...
    @direction.setter
    def direction(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"
        ],
    ) -> None: ...
    @property
    def directionalVelocity(self) -> float: ...
    @directionalVelocity.setter
    def directionalVelocity(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def isotropicVelocity(self) -> float: ...
    @isotropicVelocity.setter
    def isotropicVelocity(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def maskMaterials(self) -> list[Material]: ...
    @maskMaterials.setter
    def maskMaterials(self, arg0: collections.abc.Sequence[Material]) -> None: ...

class RayTracingParameters:
    ignoreFluxBoundaries: bool
    normalizationType: NormalizationType
    sourceDirection: ...
    useRandomSeeds: bool
    def __init__(self) -> None: ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the ray tracing parameters to a metadata dict.
        """

    def toMetaDataString(self) -> str:
        """
        Convert the ray tracing parameters to a metadata string.
        """

    @property
    def diskRadius(self) -> float: ...
    @diskRadius.setter
    def diskRadius(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def raysPerPoint(self) -> int: ...
    @raysPerPoint.setter
    def raysPerPoint(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def smoothingNeighbors(self) -> int: ...
    @smoothingNeighbors.setter
    def smoothingNeighbors(self, arg0: typing.SupportsInt) -> None: ...

class Reader:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, fileName: str) -> None: ...
    def apply(self) -> None:
        """
        Read the domain from the specified file.
        """

    def setFileName(self, arg0: str) -> None:
        """
        Set the input file name to read (should end with .vpsd).
        """

class SF6C4F8Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> PlasmaEtchingParameters: ...
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
    def __init__(self, parameters: PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> PlasmaEtchingParameters: ...
    def setParameters(self, arg0: PlasmaEtchingParameters) -> None: ...

class SF6O2Etching(ProcessModel):
    @staticmethod
    def defaultParameters() -> PlasmaEtchingParameters: ...
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
    def __init__(self, parameters: PlasmaEtchingParameters) -> None: ...
    def getParameters(self) -> PlasmaEtchingParameters: ...
    def setParameters(self, arg0: PlasmaEtchingParameters) -> None: ...

class SelectiveEpitaxy(ProcessModel):
    def __init__(
        self,
        materialRates: collections.abc.Sequence[tuple[Material, typing.SupportsFloat]],
        rate111: typing.SupportsFloat = 0.5,
        rate100: typing.SupportsFloat = 1.0,
    ) -> None: ...

class SingleParticleALD(ProcessModel):
    def __init__(
        self,
        stickingProbability: typing.SupportsFloat,
        numCycles: typing.SupportsFloat,
        growthPerCycle: typing.SupportsFloat,
        totalCycles: typing.SupportsFloat,
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
        maskMaterial: Material = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        rate: typing.SupportsFloat,
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
        maskMaterials: collections.abc.Sequence[Material],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Mapping[Material, typing.SupportsFloat],
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
    ) -> None: ...

class SphereDistribution(ProcessModel):
    @typing.overload
    def __init__(
        self,
        radius: typing.SupportsFloat,
        gridDelta: typing.SupportsFloat,
        mask: viennals3d.viennals3d.Domain,
    ) -> None: ...
    @typing.overload
    def __init__(
        self, radius: typing.SupportsFloat, gridDelta: typing.SupportsFloat
    ) -> None: ...

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

class Time:
    @staticmethod
    def convertMillisecond() -> float: ...
    @staticmethod
    def convertMinute() -> float: ...
    @staticmethod
    def convertSecond() -> float: ...
    @staticmethod
    def getInstance() -> Time: ...
    @staticmethod
    def setUnit(arg0: str) -> None: ...
    @staticmethod
    def toShortString() -> str: ...
    @staticmethod
    def toString() -> str: ...

class TimeUnit(enum.IntEnum):
    MILLISECOND: typing.ClassVar[TimeUnit]  # value = <TimeUnit.MILLISECOND: 2>
    MINUTE: typing.ClassVar[TimeUnit]  # value = <TimeUnit.MINUTE: 0>
    SECOND: typing.ClassVar[TimeUnit]  # value = <TimeUnit.SECOND: 1>
    UNDEFINED: typing.ClassVar[TimeUnit]  # value = <TimeUnit.UNDEFINED: 3>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class ToDiskMesh:
    @typing.overload
    def __init__(self, domain: Domain, mesh: viennals3d.viennals3d.Mesh) -> None: ...
    @typing.overload
    def __init__(self) -> None: ...
    def setDomain(self, arg0: Domain) -> None:
        """
        Set the domain in the mesh converter.
        """

    def setMesh(self, arg0: viennals3d.viennals3d.Mesh) -> None:
        """
        Set the mesh in the mesh converter
        """

class WetEtching(ProcessModel):
    @typing.overload
    def __init__(
        self,
        materialRates: collections.abc.Sequence[tuple[Material, typing.SupportsFloat]],
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
        materialRates: collections.abc.Sequence[tuple[Material, typing.SupportsFloat]],
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

def setNumThreads(arg0: typing.SupportsInt) -> None: ...

D: int = 3
__version__: str = '"3.7.2"'
version: str = '"3.7.2"'
