"""
2D bindings
"""
from __future__ import annotations
import collections.abc
import enum
import typing
import viennals._core
import viennals.d2
import viennaps._core
__all__: list[str] = ['AdvectionCallback', 'Anneal', 'BoxDistribution', 'CF4O2Etching', 'CSVFileProcess', 'CustomSphereDistribution', 'DamageTableModel', 'DenseCellSet', 'DirectionalProcess', 'Domain', 'DomainSetup', 'FaradayCageEtching', 'FluorocarbonEtching', 'GDSGeometry', 'GDSReader', 'GeometricTrenchDeposition', 'GeometryFactory', 'HBrO2Etching', 'ImplantDamageHobler', 'ImplantDualPearsonIV', 'ImplantPearsonIV', 'ImplantPearsonIVChanneling', 'ImplantProfileModel', 'ImplantTableModel', 'Interpolation', 'IonBeamEtching', 'IonImplantation', 'IsotropicProcess', 'MakeFin', 'MakeHole', 'MakePlane', 'MakeStack', 'MakeTrench', 'MultiParticleProcess', 'NetDoping', 'OxideRegrowth', 'Planarize', 'Process', 'ProcessModel', 'ProcessModelBase', 'RateGrid', 'Reader', 'SF6C4F8Etching', 'SF6O2Etching', 'SelectiveEpitaxy', 'SheetResistance', 'SingleParticleALD', 'SingleParticleProcess', 'SphereDistribution', 'StencilLocalLaxFriedrichsScalar', 'TEOSDeposition', 'TEOSPECVD', 'ToDiskMesh', 'VTKRenderWindow', 'WetEtching', 'Writer', 'gpu']
class AdvectionCallback:
    domain: viennaps.d2.Domain
    def __init__(self: viennaps.d2.AdvectionCallback) -> None:
        ...
    def applyPostAdvect(self: viennaps.d2.AdvectionCallback, arg0: typing.SupportsFloat) -> bool:
        ...
    def applyPreAdvect(self: viennaps.d2.AdvectionCallback, arg0: typing.SupportsFloat) -> bool:
        ...
class Anneal(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.Anneal) -> None:
        ...
    def addIsothermalStep(self: viennaps.d2.Anneal, duration: typing.SupportsFloat, temperatureK: typing.SupportsFloat) -> None:
        ...
    def addRampStep(self: viennaps.d2.Anneal, duration: typing.SupportsFloat, startT: typing.SupportsFloat, endT: typing.SupportsFloat) -> None:
        ...
    def applyActivation(self: viennaps.d2.Anneal, domain: viennaps.d2.Domain) -> None:
        """
        Apply only the solid-activation model without running diffusion.
        
        Equivalent to Sentaurus 'diffuse time=0': writes the active-
        concentration field immediately after implantation so that
        SheetResistance and NetDoping work before the full thermal anneal.
        
        Prerequisites: enableSolidActivation(True) and
        setSolidSolubilityArrhenius(C0, Ea) must be configured.
        """
    def clearDefectDiagnostics(self: viennaps.d2.Anneal) -> None:
        ...
    def clearEquilibriumArrhenius(self: viennaps.d2.Anneal) -> None:
        ...
    def clearTemperatureSchedule(self: viennaps.d2.Anneal) -> None:
        ...
    def enableDefectClustering(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def enableDefectCoupling(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def enableDefectEquilibrium(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def enableDiagnostics(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def enableSolidActivation(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def getDefectDiagnostics(self: viennaps.d2.Anneal) -> list[..., ...]:
        ...
    def resetDefectInitialization(self: viennaps.d2.Anneal) -> None:
        ...
    def setActiveLabel(self: viennaps.d2.Anneal, label: str) -> None:
        ...
    def setArrheniusParameters(self: viennaps.d2.Anneal, D0: typing.SupportsFloat, Ea_eV: typing.SupportsFloat) -> None:
        ...
    def setBlockingMaterials(self: viennaps.d2.Anneal, materials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
    def setClampNonNegative(self: viennaps.d2.Anneal, enable: bool = True) -> None:
        ...
    def setSourceField(self: viennaps.d2.Anneal, source: typing.Sequence[float]) -> None:
        ...
    def clearSourceField(self: viennaps.d2.Anneal) -> None:
        ...
    def setDamageLabels(self: viennaps.d2.Anneal, damageLabel: str, lastDamageLabel: str) -> None:
        ...
    def setDefectClusterInitFraction(self: viennaps.d2.Anneal, fraction: typing.SupportsFloat) -> None:
        ...
    def setDefectClusterKinetics(self: viennaps.d2.Anneal, kfi: typing.SupportsFloat, kfc: typing.SupportsFloat, kr: typing.SupportsFloat) -> None:
        ...
    def setDefectClusterLabel(self: viennaps.d2.Anneal, label: str) -> None:
        ...
    def setDefectDiffusivities(self: viennaps.d2.Anneal, Di: typing.SupportsFloat, Dv: typing.SupportsFloat) -> None:
        ...
    def setDefectEnhancedDiffusion(self: viennaps.d2.Anneal, tedCoefficient: typing.SupportsFloat, normalization: typing.SupportsFloat) -> None:
        ...
    def setTEDFromDamageFactor(self: viennaps.d2.Anneal, damageFactor: typing.SupportsFloat, coefficientScale: typing.SupportsFloat = 0.5, normalization: typing.SupportsFloat = 1e+20) -> None:
        ...
    def setDefectEquilibrium(self: viennaps.d2.Anneal, Ieq: typing.SupportsFloat, Veq: typing.SupportsFloat) -> None:
        ...
    def setDefectEquilibriumArrhenius(self: viennaps.d2.Anneal, interstitialC0: typing.SupportsFloat, interstitialEa_eV: typing.SupportsFloat, vacancyC0: typing.SupportsFloat, vacancyEa_eV: typing.SupportsFloat) -> None:
        ...
    def setDefectLabels(self: viennaps.d2.Anneal, interstitialLabel: str, vacancyLabel: str) -> None:
        ...
    def setDefectPartition(self: viennaps.d2.Anneal, interstitialFraction: typing.SupportsFloat, vacancyFraction: typing.SupportsFloat) -> None:
        ...
    def setDefectPartitionFactors(self: viennaps.d2.Anneal, interstitialFactor: typing.SupportsFloat, vacancyFactor: typing.SupportsFloat) -> None:
        ...
    def setDefectReactionRates(self: viennaps.d2.Anneal, kRecombination: typing.SupportsFloat, kInterstitialSink: typing.SupportsFloat, kVacancySink: typing.SupportsFloat) -> None:
        ...
    def setDefectSourceWeights(self: viennaps.d2.Anneal, historyWeight: typing.SupportsFloat, lastImpWeight: typing.SupportsFloat) -> None:
        ...
    def setDiagnosticsMaterialFilter(self: viennaps.d2.Anneal, materialId: typing.SupportsInt) -> None:
        ...
    def setDiffusionCoefficient(self: viennaps.d2.Anneal, diffCoeff: typing.SupportsFloat) -> None:
        ...
    def setDiffusionMaterials(self: viennaps.d2.Anneal, materials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
    def setDuration(self: viennaps.d2.Anneal, seconds: typing.SupportsFloat) -> None:
        ...
    def setImplicitSolverOptions(self: viennaps.d2.Anneal, maxIterations: typing.SupportsInt, relativeTolerance: typing.SupportsFloat, relaxation: typing.SupportsFloat = 1.0) -> None:
        ...
    def setMaterialLabel(self: viennaps.d2.Anneal, label: str) -> None:
        ...
    def setMode(self: viennaps.d2.Anneal, mode: viennaps._core.AnnealMode) -> None:
        ...
    def setSolidSolubilityArrhenius(self: viennaps.d2.Anneal, C0: typing.SupportsFloat, Ea_eV: typing.SupportsFloat) -> None:
        ...
    def setSpeciesLabel(self: viennaps.d2.Anneal, label: str) -> None:
        ...
    def setStabilityFactor(self: viennaps.d2.Anneal, factor: typing.SupportsFloat) -> None:
        ...
    def setTemperature(self: viennaps.d2.Anneal, temperatureK: typing.SupportsFloat) -> None:
        ...
    def setTemperatureSchedule(self: viennaps.d2.Anneal, durations: collections.abc.Sequence[typing.SupportsFloat], temperatures: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        N durations + N (isothermal) or N+1 (ramp) temperatures.
        """
    def setTimeStep(self: viennaps.d2.Anneal, dt: typing.SupportsFloat) -> None:
        ...
class BoxDistribution(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.BoxDistribution, halfAxes: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], mask: viennals.d2.Domain) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.BoxDistribution, halfAxes: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        ...
    def addMaskMaterial(self: viennaps.d2.BoxDistribution, material: viennaps._core.Material) -> None:
        ...
    def applyToSingleMaterial(self: viennaps.d2.BoxDistribution, material: viennaps._core.Material) -> None:
        ...
class CF4O2Etching(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.CF4O2Etching) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.CF4O2Etching, ionFlux: typing.SupportsFloat, etchantFlux: typing.SupportsFloat, oxygenFlux: typing.SupportsFloat, polymerFlux: typing.SupportsFloat, meanIonEnergy: typing.SupportsFloat = 100.0, sigmaIonEnergy: typing.SupportsFloat = 10.0, ionExponent: typing.SupportsFloat = 100.0, oxySputterYield: typing.SupportsFloat = 3.0, polySputterYield: typing.SupportsFloat = 3.0, etchStopDepth: typing.SupportsFloat = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.CF4O2Etching, parameters: viennaps._core.CF4O2Parameters) -> None:
        ...
    def getParameters(self: viennaps.d2.CF4O2Etching) -> viennaps._core.CF4O2Parameters:
        ...
    def setParameters(self: viennaps.d2.CF4O2Etching, arg0: viennaps._core.CF4O2Parameters) -> None:
        ...
class CSVFileProcess(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.CSVFileProcess, ratesFile: str, direction: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], offset: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"], isotropicComponent: typing.SupportsFloat = 0.0, directionalComponent: typing.SupportsFloat = 1.0, maskMaterials: collections.abc.Sequence[viennaps._core.Material] = ..., calculateVisibility: bool = True) -> None:
        ...
    def setCustomInterpolator(self: viennaps.d2.CSVFileProcess, function: collections.abc.Callable) -> None:
        ...
    def setIDWNeighbors(self: viennaps.d2.CSVFileProcess, k: typing.SupportsInt = 4) -> None:
        ...
    @typing.overload
    def setInterpolationMode(self: viennaps.d2.CSVFileProcess, mode: viennaps.d2.Interpolation) -> None:
        ...
    @typing.overload
    def setInterpolationMode(self: viennaps.d2.CSVFileProcess, mode: str) -> None:
        ...
    def setOffset(self: viennaps.d2.CSVFileProcess, offset: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"]) -> None:
        ...
class CustomSphereDistribution(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.CustomSphereDistribution, radii: collections.abc.Sequence[typing.SupportsFloat], mask: viennals.d2.Domain = None) -> None:
        ...
    def addMaskMaterial(self: viennaps.d2.CustomSphereDistribution, material: viennaps._core.Material) -> None:
        ...
class DamageTableModel(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.DamageTableModel, fileName: str, species: str, material: str, energyKeV: typing.SupportsFloat, tiltDeg: typing.SupportsFloat, rotationDeg: typing.SupportsFloat, dosePerCm2: typing.SupportsFloat = 0.0, screenThickness: typing.SupportsFloat = 0.0) -> None:
        """
        Table-backed implant-damage profile model. Pass an explicit modeldb CSV path.
        """
class DenseCellSet:
    def __init__(self: viennaps.d2.DenseCellSet) -> None:
        ...
    @typing.overload
    def addFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> bool:
        """
        Add to the filling fraction at given cell index.
        """
    @typing.overload
    def addFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.SupportsFloat) -> bool:
        """
        Add to the filling fraction for cell which contains given point.
        """
    def addFillingFractionInMaterial(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.SupportsFloat, arg2: typing.SupportsInt) -> bool:
        """
        Add to the filling fraction for cell which contains given point only if the cell has the specified material ID.
        """
    def addScalarData(self: viennaps.d2.DenseCellSet, arg0: str, arg1: typing.SupportsFloat) -> None:
        """
        Add a scalar value to be stored and modified in each cell.
        """
    def buildNeighborhood(self: viennaps.d2.DenseCellSet, forceRebuild: bool = False) -> None:
        """
        Generate fast neighbor access for each cell.
        """
    def clear(self: viennaps.d2.DenseCellSet) -> None:
        """
        Clear the filling fractions.
        """
    def fromLevelSets(self: viennaps.d2.DenseCellSet, levelSets: collections.abc.Sequence[viennals.d2.Domain], materialMap: viennals._core.MaterialMap = None, depth: typing.SupportsFloat = 0.0) -> None:
        ...
    def getAverageFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.SupportsFloat) -> float:
        """
        Get the average filling at a point in some radius.
        """
    def getBoundingBox(self: viennaps.d2.DenseCellSet) -> typing.Annotated[list[typing.Annotated[list[float], "FixedSize(2)"]], "FixedSize(2)"]:
        ...
    def getCellCenter(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Get the center of a cell with given index
        """
    def getCellGrid(self: viennaps.d2.DenseCellSet) -> viennals._core.Mesh:
        """
        Get the underlying mesh of the cell set.
        """
    def getDepth(self: viennaps.d2.DenseCellSet) -> float:
        """
        Get the depth of the cell set.
        """
    def getElement(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt) -> typing.Annotated[list[int], "FixedSize(4)"]:
        """
        Get the element at the given index.
        """
    def getElements(self: viennaps.d2.DenseCellSet) -> list[typing.Annotated[list[int], "FixedSize(4)"]]:
        """
        Get elements (cells). The indicies in the elements correspond to the corner nodes.
        """
    def getFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"]) -> float:
        """
        Get the filling fraction of the cell containing the point.
        """
    def getFillingFractions(self: viennaps.d2.DenseCellSet) -> list[float]:
        """
        Get the filling fractions of all cells.
        """
    def getGridDelta(self: viennaps.d2.DenseCellSet) -> float:
        """
        Get the cell size.
        """
    def getIndex(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> int:
        """
        Get the index of the cell containing the given point.
        """
    def getNeighbors(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt) -> typing.Annotated[list[int], "FixedSize(4)"]:
        """
        Get the neighbor indices for a cell.
        """
    def getNode(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt) -> typing.Annotated[list[float], "FixedSize(3)"]:
        """
        Get the node at the given index.
        """
    def getNodes(self: viennaps.d2.DenseCellSet) -> list[typing.Annotated[list[float], "FixedSize(3)"]]:
        """
        Get the nodes of the cell set which correspond to the corner points of the cells.
        """
    def getNumberOfCells(self: viennaps.d2.DenseCellSet) -> int:
        """
        Get the number of cells.
        """
    def getScalarData(self: viennaps.d2.DenseCellSet, arg0: str) -> list[float]:
        """
        Get the data stored at each cell. WARNING: This function only returns a copy of the data
        """
    def getScalarDataLabels(self: viennaps.d2.DenseCellSet) -> list[str]:
        """
        Get the labels of the scalar data stored in the cell set.
        """
    def getSurface(self: viennaps.d2.DenseCellSet) -> viennals.d2.Domain:
        """
        Get the surface level-set.
        """
    def readCellSetData(self: viennaps.d2.DenseCellSet, arg0: str) -> None:
        """
        Read cell set data from text.
        """
    def setCellSetPosition(self: viennaps.d2.DenseCellSet, arg0: bool) -> None:
        """
        Set whether the cell set should be created below (false) or above (true) the surface.
        """
    def setCoverMaterial(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt) -> None:
        """
        Set the material of the cells which are above or below the surface.
        """
    @typing.overload
    def setFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> bool:
        """
        Sets the filling fraction at given cell index.
        """
    @typing.overload
    def setFillingFraction(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], arg1: typing.SupportsFloat) -> bool:
        """
        Sets the filling fraction for cell which contains given point.
        """
    def setPeriodicBoundary(self: viennaps.d2.DenseCellSet, arg0: typing.Annotated[collections.abc.Sequence[bool], "FixedSize(2)"]) -> None:
        """
        Enable periodic boundary conditions in specified dimensions.
        """
    def setScalarData(self: viennaps.d2.DenseCellSet, name: str, newData: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Overwrite the scalar data associated with 'name' with a new array.
        """
    def updateMaterials(self: viennaps.d2.DenseCellSet) -> None:
        """
        Update the material IDs of the cell set. This function should be called if the level sets, the cell set is made out of, have changed. This does not work if the surface of the volume has changed. In this case, call the function 'updateSurface' first.
        """
    def updateSurface(self: viennaps.d2.DenseCellSet) -> None:
        """
        Updates the surface of the cell set. The new surface should be below the old surface as this function can only remove cells from the cell set.
        """
    def writeCellSetData(self: viennaps.d2.DenseCellSet, arg0: str) -> None:
        """
        Save cell set data in simple text format.
        """
    def writeVTU(self: viennaps.d2.DenseCellSet, arg0: str) -> None:
        """
        Write the cell set as .vtu file
        """
class DirectionalProcess(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.DirectionalProcess, direction: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], materialRates: collections.abc.Mapping[viennaps._core.Material, tuple[typing.SupportsFloat, typing.SupportsFloat]], defaultDirectionalRate: typing.SupportsFloat = 0.0, defaultIsotropicRate: typing.SupportsFloat = 0.0) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.DirectionalProcess, direction: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], directionalVelocity: typing.SupportsFloat, isotropicVelocity: typing.SupportsFloat = 0.0, maskMaterial: viennaps._core.Material = ..., calculateVisibility: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.DirectionalProcess, direction: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], directionalVelocity: typing.SupportsFloat, isotropicVelocity: typing.SupportsFloat, maskMaterial: collections.abc.Sequence[viennaps._core.Material], calculateVisibility: bool = True) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.DirectionalProcess, rateSets: collections.abc.Sequence[viennaps._core.RateSet]) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.DirectionalProcess, rateSet: viennaps._core.RateSet) -> None:
        ...
class Domain:
    @typing.overload
    def __init__(self: viennaps.d2.Domain) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Domain, domain: viennaps.d2.Domain) -> None:
        """
        Deep copy constructor.
        """
    @typing.overload
    def __init__(self: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, boundary: viennals._core.BoundaryConditionEnum = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, boundary: viennals._core.BoundaryConditionEnum = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Domain, bounds: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(4)"], boundaryConditions: typing.Annotated[collections.abc.Sequence[viennals._core.BoundaryConditionEnum], "FixedSize(2)"], gridDelta: typing.SupportsFloat = 1.0) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Domain, setup: viennaps.d2.DomainSetup) -> None:
        ...
    @typing.overload
    def addMetaData(self: viennaps.d2.Domain, arg0: str, arg1: typing.SupportsFloat) -> None:
        """
        Add a single metadata entry to the domain.
        """
    @typing.overload
    def addMetaData(self: viennaps.d2.Domain, arg0: str, arg1: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        """
        Add a single metadata entry to the domain.
        """
    @typing.overload
    def addMetaData(self: viennaps.d2.Domain, arg0: collections.abc.Mapping[str, collections.abc.Sequence[typing.SupportsFloat]]) -> None:
        """
        Add metadata to the domain.
        """
    def applyBooleanOperation(self: viennaps.d2.Domain, levelSet: viennals.d2.Domain, operation: viennals._core.BooleanOperationEnum, applyToAll: bool = True) -> None:
        """
        Apply a boolean operation with the passed Level-Set to all (or top only) Level-Sets in the domain.
        """
    def clear(self: viennaps.d2.Domain) -> None:
        ...
    def clearMetaData(self: viennaps.d2.Domain, clearDomainData: bool = False) -> None:
        """
        Clear meta data from domain.
        """
    def deepCopy(self: viennaps.d2.Domain, arg0: viennaps.d2.Domain) -> None:
        ...
    def disableMetaData(self: viennaps.d2.Domain) -> None:
        """
        Disable adding meta data to domain.
        """
    @typing.overload
    def duplicateTopLevelSet(self: viennaps.d2.Domain, arg0: viennaps._core.Material) -> None:
        """
        Duplicate the top level set. Should be used before a deposition process.
        """
    @typing.overload
    def duplicateTopLevelSet(self: viennaps.d2.Domain, arg0: str) -> None:
        """
        Duplicate the top level set. Should be used before a deposition process.
        """
    def enableMetaData(self: viennaps.d2.Domain, level: viennaps._core.MetaDataLevel = ...) -> None:
        """
        Enable adding meta data from processes to domain.
        """
    def generateCellSet(self: viennaps.d2.Domain, arg0: typing.SupportsFloat, arg1: viennaps._core.Material, arg2: bool) -> None:
        """
        Generate the cell set.
        """
    def getBoundaryConditions(self: viennaps.d2.Domain) -> typing.Annotated[list[viennals._core.BoundaryConditionEnum], "FixedSize(2)"]:
        """
        Get the boundary conditions of the domain.
        """
    def getBoundingBox(self: viennaps.d2.Domain) -> typing.Annotated[list[typing.Annotated[list[float], "FixedSize(3)"]], "FixedSize(2)"]:
        """
        Get the bounding box of the domain.
        """
    def getCellSet(self: viennaps.d2.Domain) -> viennaps.d2.DenseCellSet:
        """
        Get the cell set.
        """
    def getDiskMesh(self: viennaps.d2.Domain) -> viennals._core.Mesh:
        ...
    def getGrid(self: viennaps.d2.Domain) -> viennals.d2.hrleGrid:
        """
        Get the grid
        """
    def getGridDelta(self: viennaps.d2.Domain) -> float:
        """
        Get the grid delta.
        """
    def getHullMesh(self: viennaps.d2.Domain, bottomExtension: typing.SupportsFloat = 0.0, sharpCorners: bool = False) -> viennals._core.Mesh:
        ...
    def getLevelSetMesh(self: viennaps.d2.Domain, width: typing.SupportsInt = 1) -> list[viennals._core.Mesh]:
        """
        Get the level set grids of layers in the domain.
        """
    def getLevelSets(self: viennaps.d2.Domain) -> list[viennals.d2.Domain]:
        ...
    def getMaterialLevelSet(self: viennaps.d2.Domain, material: viennaps._core.Material) -> viennals.d2.Domain:
        """
        Returns a Level-Set representing the specified material in the domain.
        """
    def getMaterialMap(self: viennaps.d2.Domain) -> viennaps._core.MaterialMap:
        ...
    def getMaterialsInDomain(self: viennaps.d2.Domain) -> set[viennaps._core.Material]:
        """
        Get the material IDs present in the domain.
        """
    def getMetaData(self: viennaps.d2.Domain) -> dict[str, list[float]]:
        """
        Get meta data (e.g. process data) stored in the domain
        """
    def getMetaDataLevel(self: viennaps.d2.Domain) -> viennaps._core.MetaDataLevel:
        """
        Get the current meta data level of the domain.
        """
    def getNumberOfComponents(self: viennaps.d2.Domain) -> int:
        """
        Get the number of connected components in the domain.
        """
    def getNumberOfLevelSets(self: viennaps.d2.Domain) -> int:
        """
        Get the number of level sets in the domain.
        """
    def getSetup(self: viennaps.d2.Domain) -> viennaps.d2.DomainSetup:
        """
        Get the domain setup.
        """
    def getSurface(self: viennaps.d2.Domain) -> viennals.d2.Domain:
        """
        Get the surface level set.
        """
    def getSurfaceMesh(self: viennaps.d2.Domain, addInterfaces: bool = True, sharpCorners: bool = False, minNodeDistanceFactor: typing.SupportsFloat = 0.01) -> viennals._core.Mesh:
        """
        Get the surface mesh of the domain
        """
    def insertMask(self: viennaps.d2.Domain, mask: viennals.d2.Domain, material: viennaps._core.Material = ...) -> None:
        """
        Insert a mask level set to the domain. The mask is inserted at the front of the level set vector and can be used to exclude areas from processes.
        """
    @typing.overload
    def insertNextLevelSetAsMaterial(self: viennaps.d2.Domain, levelSet: viennals.d2.Domain, material: str, wrapLowerLevelSet: bool = True) -> None:
        """
        Insert a level set to domain as a material.
        """
    @typing.overload
    def insertNextLevelSetAsMaterial(self: viennaps.d2.Domain, levelSet: viennals.d2.Domain, material: viennaps._core.Material, wrapLowerLevelSet: bool = True) -> None:
        """
        Insert a level set to domain as a material.
        """
    def print(self: viennaps.d2.Domain, hrleInfo: bool = False) -> None:
        """
        Print the domain information.
        """
    def removeLevelSet(self: viennaps.d2.Domain, arg0: typing.SupportsInt, arg1: bool) -> None:
        ...
    def removeMaterial(self: viennaps.d2.Domain, arg0: viennaps._core.Material) -> None:
        ...
    def removeStrayPoints(self: viennaps.d2.Domain) -> None:
        ...
    def removeTopLevelSet(self: viennaps.d2.Domain) -> None:
        ...
    def saveDiskMesh(self: viennaps.d2.Domain, filename: str) -> None:
        ...
    def saveHullMesh(self: viennaps.d2.Domain, filename: str, bottomExtension: typing.SupportsFloat = 0.0, sharpCorners: bool = False) -> None:
        """
        Save the hull of the domain.
        """
    def saveLevelSetMesh(self: viennaps.d2.Domain, filename: str, width: typing.SupportsInt = 1) -> None:
        """
        Save the level set grids of layers in the domain.
        """
    def saveLevelSets(self: viennaps.d2.Domain, filename: str) -> None:
        ...
    def saveSurfaceMesh(self: viennaps.d2.Domain, filename: str, addInterfaces: bool = True, sharpCorners: bool = False, minNodeDistanceFactor: typing.SupportsFloat = 0.01) -> None:
        """
        Save the surface of the domain.
        """
    def saveVolumeMesh(self: viennaps.d2.Domain, filename: str, wrappingLayerEpsilon: typing.SupportsFloat = 0.01) -> None:
        """
        Save the volume representation of the domain.
        """
    def setMaterialMap(self: viennaps.d2.Domain, arg0: viennaps._core.MaterialMap) -> None:
        ...
    @typing.overload
    def setup(self: viennaps.d2.Domain, arg0: viennaps.d2.DomainSetup) -> None:
        """
        Setup the domain.
        """
    @typing.overload
    def setup(self: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat = 0.0, boundary: viennals._core.BoundaryConditionEnum = ...) -> None:
        """
        Setup the domain.
        """
    def show(self: viennaps.d2.Domain) -> None:
        """
        Render the domain using VTK.
        """
class DomainSetup:
    @typing.overload
    def __init__(self: viennaps.d2.DomainSetup) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.DomainSetup, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, boundary: viennals._core.BoundaryConditionEnum = ...) -> None:
        ...
    def boundaryCons(self: viennaps.d2.DomainSetup) -> typing.Annotated[list[viennals._core.BoundaryConditionEnum], "FixedSize(2)"]:
        ...
    def bounds(self: viennaps.d2.DomainSetup) -> typing.Annotated[list[float], "FixedSize(4)"]:
        ...
    def check(self: viennaps.d2.DomainSetup) -> None:
        ...
    def grid(self: viennaps.d2.DomainSetup) -> viennals.d2.hrleGrid:
        ...
    def gridDelta(self: viennaps.d2.DomainSetup) -> float:
        ...
    def halveXAxis(self: viennaps.d2.DomainSetup) -> None:
        ...
    def halveYAxis(self: viennaps.d2.DomainSetup) -> None:
        ...
    def hasPeriodicBoundary(self: viennaps.d2.DomainSetup) -> bool:
        ...
    def isValid(self: viennaps.d2.DomainSetup) -> bool:
        ...
    def print(self: viennaps.d2.DomainSetup) -> None:
        ...
    def xExtent(self: viennaps.d2.DomainSetup) -> float:
        ...
    def yExtent(self: viennaps.d2.DomainSetup) -> float:
        ...
class FaradayCageEtching(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.FaradayCageEtching, parameters: viennaps._core.FaradayCageParameters, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
class FluorocarbonEtching(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.FluorocarbonEtching, parameters: viennaps._core.FluorocarbonParameters) -> None:
        ...
    def setParameters(self: viennaps.d2.FluorocarbonEtching, arg0: viennaps._core.FluorocarbonParameters) -> None:
        ...
class GDSGeometry:
    @typing.overload
    def __init__(self: viennaps.d2.GDSGeometry) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.GDSGeometry, gridDelta: typing.SupportsFloat) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.GDSGeometry, gridDelta: typing.SupportsFloat, boundaryConditions: typing.Annotated[collections.abc.Sequence[viennals._core.BoundaryConditionEnum], "FixedSize(2)"]) -> None:
        ...
    def addBlur(self: viennaps.d2.GDSGeometry, sigmas: collections.abc.Sequence[typing.SupportsFloat], weights: collections.abc.Sequence[typing.SupportsFloat], threshold: typing.SupportsFloat = 0.5, delta: typing.SupportsFloat = 0.0, gridRefinement: typing.SupportsInt = 4) -> None:
        """
        Set parameters for applying mask blurring.
        """
    def getAllLayers(self: viennaps.d2.GDSGeometry) -> set[int]:
        """
        Return a set of all layers found in the GDS file.
        """
    def getBounds(self: viennaps.d2.GDSGeometry) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get the bounds of the geometry.
        """
    def getNumberOfStructures(self: viennaps.d2.GDSGeometry) -> int:
        """
        Return number of structure definitions.
        """
    def layerToLevelSet(self: viennaps.d2.GDSGeometry, layer: typing.SupportsInt, blurLayer: bool = True) -> viennals.d2.Domain:
        ...
    def print(self: viennaps.d2.GDSGeometry) -> None:
        """
        Print the geometry contents.
        """
    def setBoundaryConditions(self: viennaps.d2.GDSGeometry, arg0: collections.abc.Sequence[viennals._core.BoundaryConditionEnum]) -> None:
        """
        Set the boundary conditions
        """
    def setBoundaryPadding(self: viennaps.d2.GDSGeometry, arg0: typing.SupportsFloat, arg1: typing.SupportsFloat) -> None:
        """
        Set padding between the largest point of the geometry and the boundary of the domain.
        """
    def setGridDelta(self: viennaps.d2.GDSGeometry, arg0: typing.SupportsFloat) -> None:
        """
        Set the grid spacing.
        """
class GDSReader:
    @typing.overload
    def __init__(self: viennaps.d2.GDSReader) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.GDSReader, arg0: viennaps.d2.GDSGeometry, arg1: str) -> None:
        ...
    def apply(self: viennaps.d2.GDSReader) -> None:
        """
        Parse the GDS file.
        """
    def setFileName(self: viennaps.d2.GDSReader, arg0: str) -> None:
        """
        Set name of the GDS file.
        """
    def setGeometry(self: viennaps.d2.GDSReader, arg0: viennaps.d2.GDSGeometry) -> None:
        """
        Set the domain to be parsed in.
        """
class GeometricTrenchDeposition(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.GeometricTrenchDeposition, trenchWidth: typing.SupportsFloat, trenchDepth: typing.SupportsFloat, depositionRate: typing.SupportsFloat, bottomMed: typing.SupportsFloat, a: typing.SupportsFloat, b: typing.SupportsFloat, n: typing.SupportsFloat) -> None:
        ...
class GeometryFactory:
    def __init__(self: viennaps.d2.GeometryFactory, domainSetup: viennaps.d2.DomainSetup, name: str = 'GeometryFactory') -> None:
        ...
    def makeBoxStencil(self: viennaps.d2.GeometryFactory, position: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"], width: typing.SupportsFloat, height: typing.SupportsFloat, angle: typing.SupportsFloat = 0.0, length: typing.SupportsFloat = -1.0) -> viennals.d2.Domain:
        ...
    def makeCylinderStencil(self: viennaps.d2.GeometryFactory, position: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"], radius: typing.SupportsFloat, height: typing.SupportsFloat, angle: typing.SupportsFloat = 0.0) -> viennals.d2.Domain:
        ...
    def makeMask(self: viennaps.d2.GeometryFactory, base: typing.SupportsFloat, height: typing.SupportsFloat) -> viennals.d2.Domain:
        ...
    def makeSubstrate(self: viennaps.d2.GeometryFactory, base: typing.SupportsFloat) -> viennals.d2.Domain:
        ...
class HBrO2Etching(viennaps.d2.ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.HBrO2Etching) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.HBrO2Etching, ionFlux: typing.SupportsFloat, etchantFlux: typing.SupportsFloat, oxygenFlux: typing.SupportsFloat, meanIonEnergy: typing.SupportsFloat = 100.0, sigmaIonEnergy: typing.SupportsFloat = 10.0, ionExponent: typing.SupportsFloat = 100.0, oxySputterYield: typing.SupportsFloat = 3.0, etchStopDepth: typing.SupportsFloat = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.HBrO2Etching, parameters: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
    def getParameters(self: viennaps.d2.HBrO2Etching) -> viennaps._core.PlasmaEtchingParameters:
        ...
    def setParameters(self: viennaps.d2.HBrO2Etching, arg0: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
class ImplantDamageHobler(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.ImplantDamageHobler, projectedRange: typing.SupportsFloat, verticalSigma: typing.SupportsFloat, lambda: typing.SupportsFloat, defectsPerIon: typing.SupportsFloat, lateralSigma: typing.SupportsFloat, lateralDeltaSigma: typing.SupportsFloat = 0.0) -> None:
        """
        Hobler damage depth profile with linear-depth-scale lateral spread.
        """
class ImplantDualPearsonIV(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.ImplantDualPearsonIV, headParams: viennaps._core.PearsonIVParameters, tailParams: viennaps._core.PearsonIVParameters, headFraction: typing.SupportsFloat, headLateralMu: typing.SupportsFloat, headLateralSigma: typing.SupportsFloat, tailLateralMu: typing.SupportsFloat, tailLateralSigma: typing.SupportsFloat) -> None:
        """
        Weighted sum of two Pearson IV components (head fraction in head).
        """
class ImplantPearsonIV(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.ImplantPearsonIV, params: viennaps._core.PearsonIVParameters, lateralMu: typing.SupportsFloat, lateralSigma: typing.SupportsFloat) -> None:
        """
        Construct from PearsonIVParameters and Gaussian lateral spread.
        """
class ImplantPearsonIVChanneling(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.ImplantPearsonIVChanneling, params: viennaps._core.PearsonIVParameters, lateralMu: typing.SupportsFloat, lateralSigma: typing.SupportsFloat, tailFraction: typing.SupportsFloat, tailStartDepth: typing.SupportsFloat, tailDecayLength: typing.SupportsFloat, tailBlendWidth: typing.SupportsFloat = 0.0) -> None:
        """
        Single Pearson IV plus exponential channeling tail.
        """
class ImplantProfileModel:
    def getDepthProfile(self: viennaps.d2.ImplantProfileModel, depth: typing.SupportsFloat) -> float:
        ...
    def getLateralProfile(self: viennaps.d2.ImplantProfileModel, offset: typing.SupportsFloat, depth: typing.SupportsFloat) -> float:
        ...
    def getMaxDepth(self: viennaps.d2.ImplantProfileModel) -> float:
        ...
    def getMaxLateralRange(self: viennaps.d2.ImplantProfileModel) -> float:
        ...
class ImplantTableModel(viennaps.d2.ImplantProfileModel):
    def __init__(self: viennaps.d2.ImplantTableModel, fileName: str, species: str, material: str, substrateType: str, energyKeV: typing.SupportsFloat, tiltDeg: typing.SupportsFloat, rotationDeg: typing.SupportsFloat, dosePerCm2: typing.SupportsFloat = 0.0, screenThickness: typing.SupportsFloat = 0.0, damageLevel: typing.SupportsFloat = 0.0, preferredModel: str = 'auto') -> None:
        """
        Table-backed implant profile model. Pass an explicit modeldb CSV path; the selected row is interpolated and converted to a profile.
        """
class Interpolation(enum.IntEnum):
    CUSTOM: typing.ClassVar[viennaps.d2.Interpolation]  # value = <Interpolation.CUSTOM: 2>
    IDW: typing.ClassVar[viennaps.d2.Interpolation]  # value = <Interpolation.IDW: 1>
    LINEAR: typing.ClassVar[viennaps.d2.Interpolation]  # value = <Interpolation.LINEAR: 0>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class IonBeamEtching(viennaps.d2.ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.IBEParameters:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.IonBeamEtching, parameters: viennaps._core.IBEParameters) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.IonBeamEtching, parameters: viennaps._core.IBEParameters, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
class IonImplantation(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.IonImplantation) -> None:
        ...
    def enableBeamHits(self: viennaps.d2.IonImplantation, enable: bool = True) -> None:
        """
        Write the optional beam-hit count field.
        """
    def setBeamHitsLabel(self: viennaps.d2.IonImplantation, label: str) -> None:
        """
        Field name for optional beam-hit counts.
        """
    def setConcentrationLabel(self: viennaps.d2.IonImplantation, label: str) -> None:
        """
        Cell-set field name for deposited concentration.
        """
    def setDamageFactor(self: viennaps.d2.IonImplantation, factor: typing.SupportsFloat) -> None:
        """
        Scale factor for damage accumulation across multiple implants.
        """
    def enableEmbeddedBoundaries(self: viennaps.d2.IonImplantation, enable: bool = True) -> None:
        """
        Rebuild the cell set with embedded boundary points before implanting.
        Enables sub-grid surface offsets via ray-plane intersection for tilt accuracy.
        """
    def setDamageLabel(self: viennaps.d2.IonImplantation, label: str) -> None:
        """
        Cell-set field name for accumulated damage.
        """
    def setDamageModel(self: viennaps.d2.IonImplantation, model: viennaps.d2.ImplantProfileModel) -> None:
        """
        Set the damage profile model (optional).
        """
    def setDose(self: viennaps.d2.IonImplantation, dosePerCm2: typing.SupportsFloat) -> None:
        """
        Implant dose in ions/cm².
        """
    def setDoseControl(self: viennaps.d2.IonImplantation, mode: viennaps._core.ImplantDoseControl) -> None:
        """
        Dose control mode: Off, WaferDose, BeamDose.
        """
    def setImplantModel(self: viennaps.d2.IonImplantation, model: viennaps.d2.ImplantProfileModel) -> None:
        """
        Set the dopant concentration profile model.
        """
    def setLastDamageLabel(self: viennaps.d2.IonImplantation, label: str) -> None:
        """
        Field name for damage from the last step only.
        """
    def setLengthUnit(self: viennaps.d2.IonImplantation, lengthUnitInCm: typing.SupportsFloat) -> None:
        """
        Length unit in cm (default 1e-7 = nanometres).
        """
    def setMaskMaterials(self: viennaps.d2.IonImplantation, materials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        """
        Materials that completely block the beam.
        """
    def setOutputConcentrationInCm3(self: viennaps.d2.IonImplantation, enable: bool = True) -> None:
        """
        Store concentration in cm⁻³ instead of length-unit⁻³.
        """
    def setScreenMaterials(self: viennaps.d2.IonImplantation, materials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        """
        Materials the beam passes through without absorbing dose.
        """
    def setTiltAngle(self: viennaps.d2.IonImplantation, angleDeg: typing.SupportsFloat) -> None:
        """
        Beam tilt angle in degrees (0 = normal).
        """
    def setVoidMaterials(self: viennaps.d2.IonImplantation, materials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        """
        Materials ignored by implantation rays; usually set from the domain cover material automatically.
        """
class IsotropicProcess(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.IsotropicProcess, rate: typing.SupportsFloat = 1.0, maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.IsotropicProcess, rate: typing.SupportsFloat = 1.0, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.IsotropicProcess, materialRates: collections.abc.Mapping[viennaps._core.Material, typing.SupportsFloat], defaultRate: typing.SupportsFloat = 0.0) -> None:
        ...
    def setIsotropicRate(self: viennaps.d2.IsotropicProcess, arg0: typing.SupportsFloat) -> None:
        ...
    def setMaterialRate(self: viennaps.d2.IsotropicProcess, material: viennaps._core.Material, rate: typing.SupportsFloat) -> None:
        ...
class MakeFin:
    @typing.overload
    def __init__(self: viennaps.d2.MakeFin, domain: viennaps.d2.Domain, finWidth: typing.SupportsFloat, finHeight: typing.SupportsFloat, finTaperAngle: typing.SupportsFloat = 0.0, maskHeight: typing.SupportsFloat = 0, maskTaperAngle: typing.SupportsFloat = 0, halfFin: bool = False, material: viennaps._core.Material = ..., maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeFin, domain: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, finWidth: typing.SupportsFloat, finHeight: typing.SupportsFloat, taperAngle: typing.SupportsFloat = 0.0, baseHeight: typing.SupportsFloat = 0.0, periodicBoundary: bool = False, makeMask: bool = False, material: viennaps._core.Material = ...) -> None:
        ...
    def apply(self: viennaps.d2.MakeFin) -> None:
        """
        Create a fin geometry.
        """
class MakeHole:
    @typing.overload
    def __init__(self: viennaps.d2.MakeHole, domain: viennaps.d2.Domain, holeRadius: typing.SupportsFloat, holeDepth: typing.SupportsFloat, holeTaperAngle: typing.SupportsFloat = 0.0, maskHeight: typing.SupportsFloat = 0.0, maskTaperAngle: typing.SupportsFloat = 0.0, holeShape: viennaps._core.HoleShape = ..., material: viennaps._core.Material = ..., maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeHole, domain: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, holeRadius: typing.SupportsFloat, holeDepth: typing.SupportsFloat, taperingAngle: typing.SupportsFloat = 0.0, baseHeight: typing.SupportsFloat = 0.0, periodicBoundary: bool = False, makeMask: bool = False, material: viennaps._core.Material = ..., holeShape: viennaps._core.HoleShape = ...) -> None:
        ...
    def apply(self: viennaps.d2.MakeHole) -> None:
        """
        Create a hole geometry.
        """
class MakePlane:
    @typing.overload
    def __init__(self: viennaps.d2.MakePlane, domain: viennaps.d2.Domain, height: typing.SupportsFloat = 0.0, material: viennaps._core.Material = ..., addToExisting: bool = False) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakePlane, domain: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, height: typing.SupportsFloat = 0.0, periodicBoundary: bool = False, material: viennaps._core.Material = ...) -> None:
        ...
    def apply(self: viennaps.d2.MakePlane) -> None:
        """
        Create a plane geometry or add plane to existing geometry.
        """
class MakeStack:
    @typing.overload
    def __init__(self: viennaps.d2.MakeStack, domain: viennaps.d2.Domain, numLayers: typing.SupportsInt, layerHeight: typing.SupportsFloat, substrateHeight: typing.SupportsFloat = 0, holeRadius: typing.SupportsFloat = 0, trenchWidth: typing.SupportsFloat = 0, maskHeight: typing.SupportsFloat = 0, taperAngle: typing.SupportsFloat = 0, halfStack: bool = False, maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeStack, domain: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, numLayers: typing.SupportsInt, layerHeight: typing.SupportsFloat, substrateHeight: typing.SupportsFloat, holeRadius: typing.SupportsFloat, trenchWidth: typing.SupportsFloat, maskHeight: typing.SupportsFloat, periodicBoundary: bool = False) -> None:
        ...
    def apply(self: viennaps.d2.MakeStack) -> None:
        """
        Create a stack of alternating SiO2 and Si3N4 layers.
        """
    def getHeight(self: viennaps.d2.MakeStack) -> float:
        """
        Returns the total height of the stack.
        """
    def getTopLayer(self: viennaps.d2.MakeStack) -> int:
        """
        Returns the number of layers included in the stack
        """
class MakeTrench:
    class MaterialLayer:
        @typing.overload
        def __init__(self: viennaps.d2.MakeTrench.MaterialLayer) -> None:
            ...
        @typing.overload
        def __init__(self: viennaps.d2.MakeTrench.MaterialLayer, height: typing.SupportsFloat, width: typing.SupportsFloat, taperAngle: typing.SupportsFloat, material: viennaps._core.Material, isMask: bool) -> None:
            ...
        @property
        def height(self) -> float:
            """
            Layer thickness
            """
        @height.setter
        def height(self, arg0: typing.SupportsFloat) -> None:
            ...
        @property
        def isMask(self) -> bool:
            """
            true: apply cutout (mask behavior), false: no cutout
            """
        @isMask.setter
        def isMask(self, arg0: bool) -> None:
            ...
        @property
        def material(self) -> viennaps._core.Material:
            """
            Material type for this layer
            """
        @material.setter
        def material(self, arg0: viennaps._core.Material) -> None:
            ...
        @property
        def taperAngle(self) -> float:
            """
            Taper angle for cutout (degrees)
            """
        @taperAngle.setter
        def taperAngle(self, arg0: typing.SupportsFloat) -> None:
            ...
        @property
        def width(self) -> float:
            """
            Width of cutout for this layer
            """
        @width.setter
        def width(self, arg0: typing.SupportsFloat) -> None:
            ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeTrench, domain: viennaps.d2.Domain, trenchWidth: typing.SupportsFloat, trenchDepth: typing.SupportsFloat, trenchTaperAngle: typing.SupportsFloat = 0, maskHeight: typing.SupportsFloat = 0, maskTaperAngle: typing.SupportsFloat = 0, halfTrench: bool = False, material: viennaps._core.Material = ..., maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeTrench, domain: viennaps.d2.Domain, gridDelta: typing.SupportsFloat, xExtent: typing.SupportsFloat, yExtent: typing.SupportsFloat, trenchWidth: typing.SupportsFloat, trenchDepth: typing.SupportsFloat, taperingAngle: typing.SupportsFloat = 0.0, baseHeight: typing.SupportsFloat = 0.0, periodicBoundary: bool = False, makeMask: bool = False, material: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.MakeTrench, domain: viennaps.d2.Domain, materialLayers: collections.abc.Sequence[viennaps.d2.MakeTrench.MaterialLayer], halfTrench: bool = False) -> None:
        ...
    def apply(self: viennaps.d2.MakeTrench) -> None:
        """
        Create a trench geometry.
        """
class MultiParticleProcess(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.MultiParticleProcess) -> None:
        ...
    def addIonParticle(self: viennaps.d2.MultiParticleProcess, sourcePower: typing.SupportsFloat, thetaRMin: typing.SupportsFloat = 0.0, thetaRMax: typing.SupportsFloat = 90.0, minAngle: typing.SupportsFloat = 80.0, B_sp: typing.SupportsFloat = -1.0, meanEnergy: typing.SupportsFloat = 0.0, sigmaEnergy: typing.SupportsFloat = 0.0, thresholdEnergy: typing.SupportsFloat = 0.0, inflectAngle: typing.SupportsFloat = 0.0, n: typing.SupportsFloat = 1, label: str = 'ionFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self: viennaps.d2.MultiParticleProcess, stickingProbability: typing.SupportsFloat, label: str = 'neutralFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self: viennaps.d2.MultiParticleProcess, materialSticking: collections.abc.Mapping[viennaps._core.Material, typing.SupportsFloat], defaultStickingProbability: typing.SupportsFloat = 1.0, label: str = 'neutralFlux') -> None:
        ...
    def setRateFunction(self: viennaps.d2.MultiParticleProcess, arg0: collections.abc.Callable[[collections.abc.Sequence[typing.SupportsFloat], viennaps._core.Material], float]) -> None:
        ...
class NetDoping:
    """
    Compute net doping (Σ donors − Σ acceptors) and extract the
    metallurgical junction depth from the domain's cell set.
    
    Typical flow after implanting P and B and calling Anneal.applyActivation:
    
      nd = NetDoping()
      nd.setCellSet(domain.getCellSet())
      nd.addDonorLabel('P_active')
      nd.addAcceptorLabel('B_active')
      nd.apply()                      # writes 'net_doping' to cell set
      xj = nd.junctionDepth()         # nm — metallurgical junction depth
      print(nd.junctionCount(), 'junction(s)')
    """
    def __init__(self: viennaps.d2.NetDoping) -> None:
        ...
    def addAcceptorLabel(self: viennaps.d2.NetDoping, label: str) -> None:
        """
        Append one acceptor (p-type) concentration field name.
        """
    def addDonorLabel(self: viennaps.d2.NetDoping, label: str) -> None:
        """
        Append one donor (n-type) concentration field name.
        """
    def apply(self: viennaps.d2.NetDoping) -> None:
        """
        Compute net_doping = Σ donors − Σ acceptors and write to the output field in the cell set.
        """
    def junctionCount(self: viennaps.d2.NetDoping) -> int:
        """
        Number of metallurgical junctions in the depth profile.
        """
    def junctionDepth(self: viennaps.d2.NetDoping) -> float:
        """
        Shallowest depth [nm] where net_doping changes sign. Returns inf if no junction exists or apply() has not been called.
        """
    def junctionDepths(self: viennaps.d2.NetDoping) -> list[float]:
        """
        All junction depths [nm], sorted ascending.  Useful for retrograde profiles with multiple crossings.
        """
    def lateralJunctionPosition(self: viennaps.d2.NetDoping, atDepth: typing.SupportsFloat) -> float:
        """
        Lateral position [nm] where net_doping changes sign at the given depth.  Use for vertical (lateral) PN junctions where P and B are implanted side by side.  Returns inf if no crossing exists.
        """
    def lateralJunctionPositions(self: viennaps.d2.NetDoping, atDepth: typing.SupportsFloat) -> list[float]:
        """
        All lateral junction positions at the given depth [nm], ascending.
        """
    def setAcceptorLabels(self: viennaps.d2.NetDoping, labels: collections.abc.Sequence[str]) -> None:
        """
        Replace the full acceptor label list.
        """
    def setCellSet(self: viennaps.d2.NetDoping, cellSet: viennaps.d2.DenseCellSet) -> None:
        """
        Attach the cell set to analyse.
        """
    def setDepthAxis(self: viennaps.d2.NetDoping, axis: typing.SupportsInt) -> None:
        """
        Cell-centre axis index for depth (default: D−1).
        """
    def setDonorLabels(self: viennaps.d2.NetDoping, labels: collections.abc.Sequence[str]) -> None:
        """
        Replace the full donor label list.
        """
    def setOutputLabel(self: viennaps.d2.NetDoping, label: str) -> None:
        """
        Name of the output field written by apply() (default: 'net_doping').
        """
    def setSurfacePosition(self: viennaps.d2.NetDoping, surfacePosition: typing.SupportsFloat) -> None:
        """
        Wafer-surface coordinate along the depth axis. Depth is computed as surfacePosition minus the cell-centre coordinate.
        """
class OxideRegrowth(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.OxideRegrowth, nitrideEtchRate: typing.SupportsFloat, oxideEtchRate: typing.SupportsFloat, redepositionRate: typing.SupportsFloat, redepositionThreshold: typing.SupportsFloat, redepositionTimeInt: typing.SupportsFloat, diffusionCoefficient: typing.SupportsFloat, sinkStrength: typing.SupportsFloat, scallopVelocity: typing.SupportsFloat, centerVelocity: typing.SupportsFloat, topHeight: typing.SupportsFloat, centerWidth: typing.SupportsFloat, stabilityFactor: typing.SupportsFloat) -> None:
        ...
class Planarize:
    @typing.overload
    def __init__(self: viennaps.d2.Planarize) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Planarize, geometry: viennaps.d2.Domain, cutoffHeight: typing.SupportsFloat = 0.0) -> None:
        ...
    def apply(self: viennaps.d2.Planarize) -> None:
        """
        Apply the planarization.
        """
    def setCutoffPosition(self: viennaps.d2.Planarize, arg0: typing.SupportsFloat) -> None:
        """
        Set the cutoff height for the planarization.
        """
    def setDomain(self: viennaps.d2.Planarize, arg0: viennaps.d2.Domain) -> None:
        """
        Set the domain in the planarization.
        """
class Process:
    @typing.overload
    def __init__(self: viennaps.d2.Process) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Process, domain: viennaps.d2.Domain) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Process, domain: viennaps.d2.Domain, model: viennaps.d2.ProcessModelBase, duration: typing.SupportsFloat = 0.0) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Process, domain: viennaps.d2.Domain, model: viennaps.d2.ProcessModelBase, duration: typing.SupportsFloat = 0.0, *args) -> None:
        ...
    def apply(self: viennaps.d2.Process) -> None:
        """
        Run the process.
        """
    def calculateFlux(self: viennaps.d2.Process) -> viennals._core.Mesh:
        """
        Perform a single-pass flux calculation.
        """
    def setDomain(self: viennaps.d2.Process, arg0: viennaps.d2.Domain) -> None:
        """
        Set the process domain.
        """
    def setFluxEngineType(self: viennaps.d2.Process, arg0: viennaps._core.FluxEngineType) -> None:
        """
        Set the flux engine type (CPU or GPU).
        """
    def setIntermediateOutputPath(self: viennaps.d2.Process, path: str) -> None:
        """
        Set the path for intermediate output files during the process.
        """
    @typing.overload
    def setParameters(self: viennaps.d2.Process, parameters: viennaps._core.AdvectionParameters) -> None:
        """
        Set the advection parameters for the process.
        """
    @typing.overload
    def setParameters(self: viennaps.d2.Process, parameters: viennaps._core.RayTracingParameters) -> None:
        """
        Set the ray tracing parameters for the process.
        """
    @typing.overload
    def setParameters(self: viennaps.d2.Process, parameters: viennaps._core.CoverageParameters) -> None:
        """
        Set the coverage parameters for the process.
        """
    @typing.overload
    def setParameters(self: viennaps.d2.Process, parameters: viennaps._core.AtomicLayerProcessParameters) -> None:
        """
        Set the atomic layer parameters for the process.
        """
    def setProcessDuration(self: viennaps.d2.Process, arg0: typing.SupportsFloat) -> None:
        """
        Set the process duration.
        """
    def setProcessModel(self: viennaps.d2.Process, arg0: viennaps.d2.ProcessModelBase) -> None:
        """
        Set the process model. This has to be a pre-configured process model.
        """
class ProcessModel(viennaps.d2.ProcessModelBase):
    @staticmethod
    def setAdvectionCallback(*args, **kwargs) -> None:
        ...
    @staticmethod
    def setGeometricModel(*args, **kwargs) -> None:
        ...
    @staticmethod
    def setVelocityField(*args, **kwargs) -> None:
        ...
    def __init__(self: viennaps.d2.ProcessModel) -> None:
        ...
    def getAdvectionCallback(self: viennaps.d2.ProcessModel) -> ...:
        ...
    def getGeometricModel(self: viennaps.d2.ProcessModel) -> ...:
        ...
    def getPrimaryDirection(self: viennaps.d2.ProcessModel) -> typing.Annotated[list[float], "FixedSize(3)"] | None:
        ...
    def getProcessName(self: viennaps.d2.ProcessModel) -> str | None:
        ...
    def getSurfaceModel(self: viennaps.d2.ProcessModel) -> ...:
        ...
    def getVelocityField(self: viennaps.d2.ProcessModel) -> ...:
        ...
    def setPrimaryDirection(self: viennaps.d2.ProcessModel, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        ...
    def setProcessName(self: viennaps.d2.ProcessModel, arg0: str) -> None:
        ...
    def setSurfaceModel(self: viennaps.d2.ProcessModel, arg0: ...) -> None:
        ...
class ProcessModelBase:
    pass
class RateGrid:
    def __init__(self: viennaps.d2.RateGrid) -> None:
        ...
    def interpolate(self: viennaps.d2.RateGrid, coord: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> float:
        ...
    def loadFromCSV(self: viennaps.d2.RateGrid, filename: str) -> bool:
        ...
    def setCustomInterpolator(self: viennaps.d2.RateGrid, function: collections.abc.Callable) -> None:
        ...
    def setIDWNeighbors(self: viennaps.d2.RateGrid, k: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def setInterpolationMode(self: viennaps.d2.RateGrid, mode: viennaps.d2.Interpolation) -> None:
        ...
    @typing.overload
    def setInterpolationMode(self: viennaps.d2.RateGrid, mode: str) -> None:
        ...
    def setOffset(self: viennaps.d2.RateGrid, offset: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"]) -> None:
        ...
class Reader:
    @typing.overload
    def __init__(self: viennaps.d2.Reader) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Reader, fileName: str) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Reader, domain: viennaps.d2.Domain, fileName: str) -> None:
        ...
    def apply(self: viennaps.d2.Reader) -> None:
        """
        Read the domain from the specified file.
        """
    def setDomain(self: viennaps.d2.Reader, arg0: viennaps.d2.Domain) -> None:
        """
        Set the domain to read into.
        """
    def setFileName(self: viennaps.d2.Reader, arg0: str) -> None:
        """
        Set the input file name to read (should end with .vpsd).
        """
class SF6C4F8Etching(viennaps.d2.ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6C4F8Etching) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6C4F8Etching, ionFlux: typing.SupportsFloat, etchantFlux: typing.SupportsFloat, meanEnergy: typing.SupportsFloat, sigmaEnergy: typing.SupportsFloat, ionExponent: typing.SupportsFloat = 300.0, etchStopDepth: typing.SupportsFloat = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6C4F8Etching, parameters: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
    def getParameters(self: viennaps.d2.SF6C4F8Etching) -> viennaps._core.PlasmaEtchingParameters:
        ...
    def setParameters(self: viennaps.d2.SF6C4F8Etching, arg0: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
class SF6O2Etching(viennaps.d2.ProcessModel):
    @staticmethod
    def defaultParameters() -> viennaps._core.PlasmaEtchingParameters:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6O2Etching) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6O2Etching, ionFlux: typing.SupportsFloat, etchantFlux: typing.SupportsFloat, oxygenFlux: typing.SupportsFloat, meanIonEnergy: typing.SupportsFloat = 100.0, sigmaIonEnergy: typing.SupportsFloat = 10.0, ionExponent: typing.SupportsFloat = 100.0, oxySputterYield: typing.SupportsFloat = 3.0, etchStopDepth: typing.SupportsFloat = -1.7976931348623157e+308) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SF6O2Etching, parameters: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
    def getParameters(self: viennaps.d2.SF6O2Etching) -> viennaps._core.PlasmaEtchingParameters:
        ...
    def setParameters(self: viennaps.d2.SF6O2Etching, arg0: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
class SelectiveEpitaxy(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.SelectiveEpitaxy, rate111: typing.SupportsFloat = 0.5, rate100: typing.SupportsFloat = 1.0) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SelectiveEpitaxy, materialRates: collections.abc.Sequence[tuple[viennaps._core.Material, typing.SupportsFloat]], rate111: typing.SupportsFloat = 0.5, rate100: typing.SupportsFloat = 1.0) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SelectiveEpitaxy, nvFactors: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], rate111: typing.SupportsFloat = 0.5, rate100: typing.SupportsFloat = 1.0) -> None:
        ...
    def setMaterialRate(self: viennaps.d2.SelectiveEpitaxy, material: viennaps._core.Material, rate: typing.SupportsFloat) -> None:
        ...
class SheetResistance:
    """
    Compute sheet resistance (Rsh, Ω/□) from an active-concentration
    field stored in the domain's cell set.
    
    Default settings target ViennaPS nm-unit domains:
      length unit = 1e-7 (nm → cm),  conc unit = 1e21 (nm⁻³ → cm⁻³),
      depth axis  = D−1  (y for 2-D, z for 3-D),
      surface position = 0  (depth = surface − coordinate).
    
    Example::
    
      sr = SheetResistance()
      sr.setCellSet(domain.getCellSet())
      sr.setConcentrationLabel("P_active")
      rsh = sr.computeElectron()   # Masetti n-type (P in Si)
    """
    def __init__(self: viennaps.d2.SheetResistance) -> None:
        ...
    def computeElectron(self: viennaps.d2.SheetResistance) -> float:
        """
        Rsh [Ω/□] using the Masetti-Severi electron mobility model (n-type, e.g. P-doped Si).
        """
    def computeHole(self: viennaps.d2.SheetResistance) -> float:
        """
        Rsh [Ω/□] using the Masetti-Severi hole mobility model (p-type, e.g. B-doped Si).
        """
    def setCellSet(self: viennaps.d2.SheetResistance, cellSet: viennaps.d2.DenseCellSet) -> None:
        """
        Attach the cell set to analyse.
        """
    def setConcentrationLabel(self: viennaps.d2.SheetResistance, label: str) -> None:
        """
        Name of the scalar field containing the active concentration (default: 'active_concentration').
        """
    def setConcentrationUnit(self: viennaps.d2.SheetResistance, unit: typing.SupportsFloat) -> None:
        """
        Multiplicative factor to convert the cell-set concentration to cm⁻³ (default: 1e21 for nm⁻³ fields).
        """
    def setDepthAxis(self: viennaps.d2.SheetResistance, axis: typing.SupportsInt) -> None:
        """
        Cell-centre axis index for depth  (default: D−1).
        """
    def setLengthUnit(self: viennaps.d2.SheetResistance, lu_cm: typing.SupportsFloat) -> None:
        """
        Length-unit → cm conversion factor (default: 1e-7 for nm domains). Also updates the concentration unit to stay consistent.
        """
    def setSurfacePosition(self: viennaps.d2.SheetResistance, surfacePosition: typing.SupportsFloat) -> None:
        """
        Wafer-surface coordinate along the depth axis. Depth is computed as surfacePosition minus the cell-centre coordinate.
        """
class SingleParticleALD(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.SingleParticleALD, stickingProbability: typing.SupportsFloat, numCycles: typing.SupportsInt, growthPerCycle: typing.SupportsFloat, totalCycles: typing.SupportsInt, coverageTimeStep: typing.SupportsFloat, evFlux: typing.SupportsFloat, inFlux: typing.SupportsFloat, s0: typing.SupportsFloat, gasMFP: typing.SupportsFloat) -> None:
        ...
class SingleParticleProcess(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.SingleParticleProcess, rate: typing.SupportsFloat = 1.0, stickingProbability: typing.SupportsFloat = 1.0, sourceExponent: typing.SupportsFloat = 1.0, maskMaterial: viennaps._core.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SingleParticleProcess, rate: typing.SupportsFloat, stickingProbability: typing.SupportsFloat, sourceExponent: typing.SupportsFloat, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SingleParticleProcess, materialRates: collections.abc.Mapping[viennaps._core.Material, typing.SupportsFloat], stickingProbability: typing.SupportsFloat, sourceExponent: typing.SupportsFloat) -> None:
        ...
    def setDefaultRate(self: viennaps.d2.SingleParticleProcess, arg0: typing.SupportsFloat) -> None:
        ...
    def setMaterialRate(self: viennaps.d2.SingleParticleProcess, material: viennaps._core.Material, rate: typing.SupportsFloat) -> None:
        ...
class SphereDistribution(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.SphereDistribution, radius: typing.SupportsFloat, mask: viennals.d2.Domain) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.SphereDistribution, radius: typing.SupportsFloat) -> None:
        ...
    def addMaskMaterial(self: viennaps.d2.SphereDistribution, material: viennaps._core.Material) -> None:
        ...
    def applyToSingleMaterial(self: viennaps.d2.SphereDistribution, material: viennaps._core.Material) -> None:
        ...
class StencilLocalLaxFriedrichsScalar:
    @staticmethod
    def setMaxDissipation(maxDissipation: typing.SupportsFloat) -> None:
        ...
class TEOSDeposition(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.TEOSDeposition, stickingProbabilityP1: typing.SupportsFloat, rateP1: typing.SupportsFloat, orderP1: typing.SupportsFloat, stickingProbabilityP2: typing.SupportsFloat = 0.0, rateP2: typing.SupportsFloat = 0.0, orderP2: typing.SupportsFloat = 0.0) -> None:
        ...
class TEOSPECVD(viennaps.d2.ProcessModel):
    def __init__(self: viennaps.d2.TEOSPECVD, stickingProbabilityRadical: typing.SupportsFloat, depositionRateRadical: typing.SupportsFloat, depositionRateIon: typing.SupportsFloat, exponentIon: typing.SupportsFloat, stickingProbabilityIon: typing.SupportsFloat = 1.0, reactionOrderRadical: typing.SupportsFloat = 1.0, reactionOrderIon: typing.SupportsFloat = 1.0, minAngleIon: typing.SupportsFloat = 0.0) -> None:
        ...
class ToDiskMesh:
    @typing.overload
    def __init__(self: viennaps.d2.ToDiskMesh, domain: viennaps.d2.Domain, mesh: viennals._core.Mesh) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.ToDiskMesh) -> None:
        ...
    def apply(self: viennaps.d2.ToDiskMesh) -> None:
        ...
    def setDomain(self: viennaps.d2.ToDiskMesh, arg0: viennaps.d2.Domain) -> None:
        """
        Set the domain in the mesh converter.
        """
    def setMesh(self: viennaps.d2.ToDiskMesh, arg0: viennals._core.Mesh) -> None:
        """
        Set the mesh in the mesh converter
        """
class VTKRenderWindow:
    @typing.overload
    def __init__(self: viennaps.d2.VTKRenderWindow) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.VTKRenderWindow, domain: viennaps.d2.Domain) -> None:
        ...
    def insertNextDomain(self: viennaps.d2.VTKRenderWindow, domain: viennaps.d2.Domain, offset: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"] = [0.0, 0.0, 0.0]) -> None:
        """
        Insert domain to be visualized.
        """
    def printCameraInfo(self: viennaps.d2.VTKRenderWindow) -> None:
        """
        Print the current camera settings to the console.
        """
    def render(self: viennaps.d2.VTKRenderWindow) -> None:
        """
        Render the current domain state.
        """
    def saveScreenshot(self: viennaps.d2.VTKRenderWindow, fileName: str, scale: typing.SupportsInt = 1) -> None:
        """
        Save a screenshot of the current render window.
        """
    def setBackgroundColor(self: viennaps.d2.VTKRenderWindow, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        """
        Set the background color of the render window.
        """
    def setCameraFocalPoint(self: viennaps.d2.VTKRenderWindow, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        """
        Set the camera focal point in world coordinates.
        """
    def setCameraPosition(self: viennaps.d2.VTKRenderWindow, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        """
        Set the camera position in world coordinates.
        """
    def setCameraView(self: viennaps.d2.VTKRenderWindow, axis: typing.SupportsInt) -> None:
        """
        Set the camera view along an axix (x,y,z)
        """
    def setCameraViewUp(self: viennaps.d2.VTKRenderWindow, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        """
        Set the camera view up vector.
        """
    def setDomainOffset(self: viennaps.d2.VTKRenderWindow, arg0: typing.SupportsInt, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"]) -> None:
        """
        Set an offset to be applied to the domain during rendering.
        """
    def setRenderMode(self: viennaps.d2.VTKRenderWindow, arg0: viennaps._core.RenderMode) -> None:
        """
        Set the render mode (surface, interfaces, volume).
        """
    def setWindowSize(self: viennaps.d2.VTKRenderWindow, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsInt], "FixedSize(2)"]) -> None:
        """
        Set the size of the render window.
        """
    def toggleInstructionText(self: viennaps.d2.VTKRenderWindow) -> None:
        """
        Toggle the instruction text overlay on/off.
        """
class WetEtching(viennaps.d2.ProcessModel):
    @typing.overload
    def __init__(self: viennaps.d2.WetEtching, materialRates: collections.abc.Sequence[tuple[viennaps._core.Material, typing.SupportsFloat]]) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.WetEtching, direction100: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], direction010: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat], "FixedSize(3)"], rate100: typing.SupportsFloat, rate110: typing.SupportsFloat, rate111: typing.SupportsFloat, rate311: typing.SupportsFloat, materialRates: collections.abc.Sequence[tuple[viennaps._core.Material, typing.SupportsFloat]]) -> None:
        ...
class Writer:
    @typing.overload
    def __init__(self: viennaps.d2.Writer) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Writer, domain: viennaps.d2.Domain) -> None:
        ...
    @typing.overload
    def __init__(self: viennaps.d2.Writer, domain: viennaps.d2.Domain, fileName: str) -> None:
        ...
    def apply(self: viennaps.d2.Writer) -> None:
        """
        Write the domain to the specified file.
        """
    def setDomain(self: viennaps.d2.Writer, arg0: viennaps.d2.Domain) -> None:
        """
        Set the domain to be written to a file.
        """
    def setFileName(self: viennaps.d2.Writer, arg0: str) -> None:
        """
        Set the output file name (should end with .vpsd).
        """
gpu = viennaps.d2.gpu
