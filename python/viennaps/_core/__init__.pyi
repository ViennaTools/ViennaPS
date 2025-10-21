"""
ViennaPS is a header-only C++ process simulation library which includes surface and volume representations, a ray tracer, and physical models for the simulation of microelectronic fabrication processes. The main design goals are simplicity and efficiency, tailored towards scientific simulations.
"""

from __future__ import annotations
import collections.abc
import enum
import typing
import viennals._core
import viennaps.d2
from viennaps import d2
from viennaps import d3
import viennaps.d3
from . import constants
from . import gpu
from . import util

__all__: list[str] = [
    "AdvectionParameters",
    "AtomicLayerProcessParameters",
    "CF4O2Parameters",
    "CF4O2ParametersIons",
    "CF4O2ParametersMask",
    "CF4O2ParametersPassivation",
    "CF4O2ParametersSi",
    "CF4O2ParametersSiGe",
    "CoverageParameters",
    "Extrude",
    "FaradayCageParameters",
    "FluorocarbonMaterialParameters",
    "FluorocarbonParameters",
    "FluorocarbonParametersIons",
    "FluxEngineType",
    "HoleShape",
    "IBEParameters",
    "Length",
    "LengthUnit",
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
    "Slice",
    "Time",
    "TimeUnit",
    "constants",
    "d2",
    "d3",
    "gpu",
    "gpuAvailable",
    "setNumThreads",
    "util",
    "version",
]

class AdvectionParameters:
    checkDissipation: bool
    ignoreVoids: bool
    integrationScheme: viennals._core.IntegrationSchemeEnum
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

class AtomicLayerProcessParameters:
    def __init__(self) -> None: ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the ALD process parameters to a metadata dict.
        """

    def toMetaDataString(self) -> str:
        """
        Convert the ALD process parameters to a metadata string.
        """

    @property
    def coverageTimeStep(self) -> float: ...
    @coverageTimeStep.setter
    def coverageTimeStep(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def numCycles(self) -> int: ...
    @numCycles.setter
    def numCycles(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def pulseTime(self) -> float: ...
    @pulseTime.setter
    def pulseTime(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def purgePulseTime(self) -> float: ...
    @purgePulseTime.setter
    def purgePulseTime(self, arg0: typing.SupportsFloat) -> None: ...

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

class CoverageParameters:
    def __init__(self) -> None: ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the coverage parameters to a metadata dict.
        """

    def toMetaDataString(self) -> str:
        """
        Convert the coverage parameters to a metadata string.
        """

    @property
    def maxIterations(self) -> int: ...
    @maxIterations.setter
    def maxIterations(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def tolerance(self) -> float: ...
    @tolerance.setter
    def tolerance(self, arg0: typing.SupportsFloat) -> None: ...

class Extrude:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        inputDomain: viennaps.d2.Domain,
        outputDomain: viennaps.d3.Domain,
        extent: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
        extrusionAxis: typing.SupportsInt,
        boundaryConditions: typing.Annotated[
            collections.abc.Sequence[viennals._core.BoundaryConditionEnum],
            "FixedSize(3)",
        ],
    ) -> None: ...
    def apply(self) -> None:
        """
        Run the extrusion.
        """

    def setBoundaryConditions(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[viennals._core.BoundaryConditionEnum],
            "FixedSize(3)",
        ],
    ) -> None:
        """
        Set the boundary conditions in the extruded domain.
        """

    def setExtent(
        self,
        arg0: typing.Annotated[
            collections.abc.Sequence[typing.SupportsFloat], "FixedSize(2)"
        ],
    ) -> None:
        """
        Set the min and max extent in the extruded dimension.
        """

    def setExtrusionAxis(self, arg0: typing.SupportsInt) -> None:
        """
        Set the axis along which to extrude (0, 1, or 2).
        """

    def setInputDomain(self, arg0: viennaps.d2.Domain) -> None:
        """
        Set the input domain to be extruded.
        """

    def setOutputDomain(self, arg0: viennaps.d3.Domain) -> None:
        """
        Set the output domain. The 3D output domain will be overwritten by the extruded domain.
        """

class FaradayCageParameters:
    ibeParams: IBEParameters
    def __init__(self) -> None: ...
    @property
    def cageAngle(self) -> float: ...
    @cageAngle.setter
    def cageAngle(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonMaterialParameters:
    id: Material
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
    def beta_e(self) -> float: ...
    @beta_e.setter
    def beta_e(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beta_p(self) -> float: ...
    @beta_p.setter
    def beta_p(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def density(self) -> float: ...
    @density.setter
    def density(self, arg0: typing.SupportsFloat) -> None: ...

class FluorocarbonParameters:
    Ions: FluorocarbonParametersIons
    def __init__(self) -> None: ...
    def addMaterial(
        self, materialParameters: FluorocarbonMaterialParameters
    ) -> None: ...
    def getMaterialParameters(
        self, material: Material
    ) -> FluorocarbonMaterialParameters: ...
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
    def k_ev(self) -> float: ...
    @k_ev.setter
    def k_ev(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def k_ie(self) -> float: ...
    @k_ie.setter
    def k_ie(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def polyFlux(self) -> float: ...
    @polyFlux.setter
    def polyFlux(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def temperature(self) -> float: ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat) -> None: ...

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

class FluxEngineType(enum.IntEnum):
    AUTO: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.AUTO: 0>
    CPU_DISK: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.CPU_DISK: 1>
    GPU_DISK: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.GPU_DISK: 3>
    GPU_LINE: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.GPU_LINE: 4>
    GPU_TRIANGLE: typing.ClassVar[
        FluxEngineType
    ]  # value = <FluxEngineType.GPU_TRIANGLE: 2>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

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
    def setLogLevel(arg0: ...) -> None: ...
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

class Material(enum.IntEnum):
    """
    Material types for domain and level sets
    """

    ARC: typing.ClassVar[Material]  # value = <Material.ARC: 57>
    AZO: typing.ClassVar[Material]  # value = <Material.AZO: 152>
    Air: typing.ClassVar[Material]  # value = <Material.Air: 2>
    Al2O3: typing.ClassVar[Material]  # value = <Material.Al2O3: 31>
    AlN: typing.ClassVar[Material]  # value = <Material.AlN: 37>
    BN: typing.ClassVar[Material]  # value = <Material.BN: 39>
    BPSG: typing.ClassVar[Material]  # value = <Material.BPSG: 54>
    C: typing.ClassVar[Material]  # value = <Material.C: 50>
    Co: typing.ClassVar[Material]  # value = <Material.Co: 72>
    CoW: typing.ClassVar[Material]  # value = <Material.CoW: 85>
    Cu: typing.ClassVar[Material]  # value = <Material.Cu: 71>
    Dielectric: typing.ClassVar[Material]  # value = <Material.Dielectric: 4>
    GAS: typing.ClassVar[Material]  # value = <Material.GAS: 3>
    GST: typing.ClassVar[Material]  # value = <Material.GST: 135>
    GaAs: typing.ClassVar[Material]  # value = <Material.GaAs: 112>
    GaN: typing.ClassVar[Material]  # value = <Material.GaN: 111>
    Ge: typing.ClassVar[Material]  # value = <Material.Ge: 110>
    Graphene: typing.ClassVar[Material]  # value = <Material.Graphene: 130>
    HSQ: typing.ClassVar[Material]  # value = <Material.HSQ: 60>
    HfO2: typing.ClassVar[Material]  # value = <Material.HfO2: 32>
    ITO: typing.ClassVar[Material]  # value = <Material.ITO: 150>
    InGaAs: typing.ClassVar[Material]  # value = <Material.InGaAs: 114>
    InP: typing.ClassVar[Material]  # value = <Material.InP: 113>
    Ir: typing.ClassVar[Material]  # value = <Material.Ir: 81>
    La2O3: typing.ClassVar[Material]  # value = <Material.La2O3: 36>
    Mask: typing.ClassVar[Material]  # value = <Material.Mask: 0>
    Metal: typing.ClassVar[Material]  # value = <Material.Metal: 5>
    Mn: typing.ClassVar[Material]  # value = <Material.Mn: 88>
    MnN: typing.ClassVar[Material]  # value = <Material.MnN: 90>
    MnO: typing.ClassVar[Material]  # value = <Material.MnO: 89>
    Mo: typing.ClassVar[Material]  # value = <Material.Mo: 80>
    MoS2: typing.ClassVar[Material]  # value = <Material.MoS2: 131>
    MoSi2: typing.ClassVar[Material]  # value = <Material.MoSi2: 102>
    Ni: typing.ClassVar[Material]  # value = <Material.Ni: 74>
    NiW: typing.ClassVar[Material]  # value = <Material.NiW: 86>
    PHS: typing.ClassVar[Material]  # value = <Material.PHS: 59>
    PMMA: typing.ClassVar[Material]  # value = <Material.PMMA: 58>
    PSG: typing.ClassVar[Material]  # value = <Material.PSG: 55>
    Pd: typing.ClassVar[Material]  # value = <Material.Pd: 83>
    PolySi: typing.ClassVar[Material]  # value = <Material.PolySi: 11>
    Polymer: typing.ClassVar[Material]  # value = <Material.Polymer: 1>
    Pt: typing.ClassVar[Material]  # value = <Material.Pt: 75>
    Rh: typing.ClassVar[Material]  # value = <Material.Rh: 82>
    Ru: typing.ClassVar[Material]  # value = <Material.Ru: 73>
    RuTa: typing.ClassVar[Material]  # value = <Material.RuTa: 84>
    SOC: typing.ClassVar[Material]  # value = <Material.SOC: 52>
    SOG: typing.ClassVar[Material]  # value = <Material.SOG: 53>
    Si: typing.ClassVar[Material]  # value = <Material.Si: 10>
    Si3N4: typing.ClassVar[Material]  # value = <Material.Si3N4: 16>
    SiBCN: typing.ClassVar[Material]  # value = <Material.SiBCN: 19>
    SiC: typing.ClassVar[Material]  # value = <Material.SiC: 14>
    SiCN: typing.ClassVar[Material]  # value = <Material.SiCN: 18>
    SiCOH: typing.ClassVar[Material]  # value = <Material.SiCOH: 20>
    SiC_HM: typing.ClassVar[Material]  # value = <Material.SiC_HM: 172>
    SiGaN: typing.ClassVar[Material]  # value = <Material.SiGaN: 115>
    SiGe: typing.ClassVar[Material]  # value = <Material.SiGe: 13>
    SiLK: typing.ClassVar[Material]  # value = <Material.SiLK: 56>
    SiN: typing.ClassVar[Material]  # value = <Material.SiN: 15>
    SiN_HM: typing.ClassVar[Material]  # value = <Material.SiN_HM: 171>
    SiO2: typing.ClassVar[Material]  # value = <Material.SiO2: 30>
    SiO2_HM: typing.ClassVar[Material]  # value = <Material.SiO2_HM: 175>
    SiOCH: typing.ClassVar[Material]  # value = <Material.SiOCH: 116>
    SiOCN: typing.ClassVar[Material]  # value = <Material.SiOCN: 21>
    SiON: typing.ClassVar[Material]  # value = <Material.SiON: 17>
    SiON_HM: typing.ClassVar[Material]  # value = <Material.SiON_HM: 170>
    Ta: typing.ClassVar[Material]  # value = <Material.Ta: 76>
    Ta2O5: typing.ClassVar[Material]  # value = <Material.Ta2O5: 38>
    TaN: typing.ClassVar[Material]  # value = <Material.TaN: 77>
    Ti: typing.ClassVar[Material]  # value = <Material.Ti: 78>
    TiAlN: typing.ClassVar[Material]  # value = <Material.TiAlN: 87>
    TiN: typing.ClassVar[Material]  # value = <Material.TiN: 79>
    TiO: typing.ClassVar[Material]  # value = <Material.TiO: 173>
    TiO2: typing.ClassVar[Material]  # value = <Material.TiO2: 34>
    TiSi2: typing.ClassVar[Material]  # value = <Material.TiSi2: 101>
    Undefined: typing.ClassVar[Material]  # value = <Material.Undefined: 6>
    VO2: typing.ClassVar[Material]  # value = <Material.VO2: 134>
    W: typing.ClassVar[Material]  # value = <Material.W: 70>
    WS2: typing.ClassVar[Material]  # value = <Material.WS2: 132>
    WSe2: typing.ClassVar[Material]  # value = <Material.WSe2: 133>
    WSi2: typing.ClassVar[Material]  # value = <Material.WSi2: 100>
    Y2O3: typing.ClassVar[Material]  # value = <Material.Y2O3: 35>
    ZnO: typing.ClassVar[Material]  # value = <Material.ZnO: 151>
    ZrO: typing.ClassVar[Material]  # value = <Material.ZrO: 174>
    ZrO2: typing.ClassVar[Material]  # value = <Material.ZrO2: 33>
    aC: typing.ClassVar[Material]  # value = <Material.aC: 51>
    aSi: typing.ClassVar[Material]  # value = <Material.aSi: 12>
    hBN: typing.ClassVar[Material]  # value = <Material.hBN: 40>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """

class MaterialMap:
    @staticmethod
    def isMaterial(arg0: typing.SupportsFloat, arg1: Material) -> bool: ...
    @staticmethod
    def mapToMaterial(arg0: typing.SupportsFloat) -> Material:
        """
        Map a float to a material.
        """

    @staticmethod
    def toString(arg0: Material) -> str:
        """
        Get the name of a material.
        """

    def __init__(self) -> None: ...
    def getMaterialAtIdx(self, arg0: typing.SupportsInt) -> Material: ...
    def getMaterialMap(self) -> viennals._core.MaterialMap: ...
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

class NormalizationType(enum.IntEnum):
    MAX: typing.ClassVar[NormalizationType]  # value = <NormalizationType.MAX: 1>
    SOURCE: typing.ClassVar[NormalizationType]  # value = <NormalizationType.SOURCE: 0>
    @classmethod
    def __new__(cls, value): ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
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

class Slice:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        inputDomain: viennaps.d3.Domain,
        outputDomain: viennaps.d2.Domain,
        sliceDimension: typing.SupportsInt,
        slicePosition: typing.SupportsFloat,
    ) -> None: ...
    def apply(self) -> None:
        """
        Run the slicing.
        """

    def setInputDomain(self, arg0: viennaps.d3.Domain) -> None:
        """
        Set the input domain to be sliced.
        """

    def setOutputDomain(self, arg0: viennaps.d2.Domain) -> None:
        """
        Set the output domain. The 2D output domain will be overwritten by the sliced domain.
        """

    def setReflectX(self, arg0: bool) -> None:
        """
        Set whether to reflect the slice along the X axis.
        """

    def setSliceDimension(self, arg0: typing.SupportsInt) -> None:
        """
        Set the dimension along which to slice (0, 1).
        """

    def setSlicePosition(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the position along the slice dimension at which to slice.
        """

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

def gpuAvailable() -> bool:
    """
    Check if ViennaPS was compiled with GPU support.
    """

def setNumThreads(arg0: typing.SupportsInt) -> None: ...

__version__: str = "4.0.0"
version: str = "4.0.0"
