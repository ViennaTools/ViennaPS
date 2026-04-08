"""
ViennaPS is a topography simulation library for microelectronic fabrication processes. It models the evolution of 2D and 3D surfaces during etching, deposition, and related steps, combining advanced level-set methods for surface evolution with Monte Carlo ray tracing for flux calculation. This allows accurate, feature-scale simulation of complex fabrication geometries.
"""
from __future__ import annotations
import collections.abc
import enum
import typing
import viennals._core
import viennaps.d2
from viennaps import d2
import viennaps.d3
from viennaps import d3
from . import constants
from . import gpu
from . import util
__all__: list[str] = ['AdvectionParameters', 'AtomicLayerProcessParameters', 'BuiltInMaterial', 'CF4O2Parameters', 'CF4O2ParametersIons', 'CF4O2ParametersMask', 'CF4O2ParametersPassivation', 'CF4O2ParametersSi', 'CF4O2ParametersSiGe', 'CoverageParameters', 'Extrude', 'FaradayCageParameters', 'FluorocarbonMaterialParameters', 'FluorocarbonParameters', 'FluorocarbonParametersIons', 'FluxEngineType', 'HoleShape', 'IBEParameters', 'IBEParametersCos4Yield', 'Length', 'LengthUnit', 'Logger', 'Material', 'MaterialCategory', 'MaterialInfo', 'MaterialKind', 'MaterialMap', 'MaterialRegistry', 'MaterialValueMap', 'MetaDataLevel', 'NormalizationType', 'PlasmaEtchingParameters', 'PlasmaEtchingParametersIons', 'PlasmaEtchingParametersMask', 'PlasmaEtchingParametersPassivation', 'PlasmaEtchingParametersPolymer', 'PlasmaEtchingParametersSubstrate', 'ProcessParams', 'RateSet', 'RayTracingParameters', 'RenderMode', 'Slice', 'Time', 'TimeUnit', 'constants', 'd2', 'd3', 'gpu', 'gpuAvailable', 'setNumThreads', 'util', 'version']
class AdvectionParameters:
    adaptiveTimeStepping: bool
    calculateIntermediateVelocities: bool
    checkDissipation: bool
    ignoreVoids: bool
    integrationScheme: viennals._core.SpatialSchemeEnum
    spatialScheme: viennals._core.SpatialSchemeEnum
    temporalScheme: viennals._core.TemporalSchemeEnum
    velocityOutput: bool
    def __init__(self) -> None:
        ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the advection parameters to a metadata dict.
        """
    def toMetaDataString(self) -> str:
        """
        Convert the advection parameters to a metadata string.
        """
    @property
    def adaptiveTimeStepSubdivisions(self) -> int:
        ...
    @adaptiveTimeStepSubdivisions.setter
    def adaptiveTimeStepSubdivisions(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def dissipationAlpha(self) -> float:
        ...
    @dissipationAlpha.setter
    def dissipationAlpha(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def timeStepRatio(self) -> float:
        ...
    @timeStepRatio.setter
    def timeStepRatio(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class AtomicLayerProcessParameters:
    def __init__(self) -> None:
        ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the ALD process parameters to a metadata dict.
        """
    def toMetaDataString(self) -> str:
        """
        Convert the ALD process parameters to a metadata string.
        """
    @property
    def coverageTimeStep(self) -> float:
        ...
    @coverageTimeStep.setter
    def coverageTimeStep(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def numCycles(self) -> int:
        ...
    @numCycles.setter
    def numCycles(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def pulseTime(self) -> float:
        ...
    @pulseTime.setter
    def pulseTime(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def purgePulseTime(self) -> float:
        ...
    @purgePulseTime.setter
    def purgePulseTime(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class BuiltInMaterial(enum.IntEnum):
    """
    Fixed built-in material types for domain and level sets
    """
    ARC: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.ARC: 57>
    AZO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.AZO: 152>
    Air: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Air: 2>
    Al2O3: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Al2O3: 31>
    AlN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.AlN: 37>
    Au: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Au: 91>
    BN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.BN: 39>
    BPSG: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.BPSG: 54>
    BulkSi: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.BulkSi: 22>
    C: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.C: 50>
    Co: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Co: 72>
    CoW: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.CoW: 85>
    Cr: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Cr: 92>
    Cu: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Cu: 71>
    Custom: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Custom: 176>
    Dielectric: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Dielectric: 4>
    GAS: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.GAS: 3>
    GST: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.GST: 135>
    GaAs: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.GaAs: 112>
    GaN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.GaN: 111>
    Ge: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ge: 110>
    Graphene: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Graphene: 130>
    HSQ: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.HSQ: 60>
    HfO2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.HfO2: 32>
    ITO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.ITO: 150>
    InGaAs: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.InGaAs: 114>
    InP: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.InP: 113>
    Ir: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ir: 81>
    La2O3: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.La2O3: 36>
    Mask: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Mask: 0>
    Metal: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Metal: 5>
    Mn: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Mn: 88>
    MnN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.MnN: 90>
    MnO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.MnO: 89>
    Mo: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Mo: 80>
    MoS2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.MoS2: 131>
    MoSi2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.MoSi2: 102>
    Ni: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ni: 74>
    NiW: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.NiW: 86>
    PHS: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.PHS: 59>
    PMMA: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.PMMA: 58>
    PSG: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.PSG: 55>
    Pd: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Pd: 83>
    PolySi: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.PolySi: 11>
    Polymer: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Polymer: 1>
    Pt: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Pt: 75>
    Rh: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Rh: 82>
    Ru: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ru: 73>
    RuTa: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.RuTa: 84>
    SOC: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SOC: 52>
    SOG: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SOG: 53>
    Si: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Si: 10>
    Si3N4: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Si3N4: 16>
    SiBCN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiBCN: 19>
    SiC: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiC: 14>
    SiCN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiCN: 18>
    SiCOH: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiCOH: 20>
    SiC_HM: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiC_HM: 172>
    SiGaN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiGaN: 115>
    SiGe: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiGe: 13>
    SiLK: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiLK: 56>
    SiN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiN: 15>
    SiN_HM: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiN_HM: 171>
    SiO2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiO2: 30>
    SiO2_HM: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiO2_HM: 175>
    SiOCH: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiOCH: 116>
    SiOCN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiOCN: 21>
    SiON: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiON: 17>
    SiON_HM: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.SiON_HM: 170>
    Ta: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ta: 76>
    Ta2O5: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ta2O5: 38>
    TaN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TaN: 77>
    Ti: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Ti: 78>
    TiAlN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TiAlN: 87>
    TiN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TiN: 79>
    TiO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TiO: 173>
    TiO2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TiO2: 34>
    TiSi2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.TiSi2: 101>
    Undefined: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Undefined: 6>
    VO2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.VO2: 134>
    W: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.W: 70>
    WS2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.WS2: 132>
    WSe2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.WSe2: 133>
    WSi2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.WSi2: 100>
    Y2O3: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.Y2O3: 35>
    ZnO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.ZnO: 151>
    ZrO: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.ZrO: 174>
    ZrO2: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.ZrO2: 33>
    aC: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.aC: 51>
    aSi: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.aSi: 12>
    hBN: typing.ClassVar[BuiltInMaterial]  # value = <BuiltInMaterial.hBN: 40>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class CF4O2Parameters:
    Ions: CF4O2ParametersIons
    Mask: CF4O2ParametersMask
    Passivation: CF4O2ParametersPassivation
    Si: CF4O2ParametersSi
    SiGe: CF4O2ParametersSiGe
    fluxIncludeSticking: bool
    gamma_C: MaterialValueMap
    gamma_C_oxidized: MaterialValueMap
    gamma_F: MaterialValueMap
    gamma_F_oxidized: MaterialValueMap
    gamma_O: MaterialValueMap
    gamma_O_passivated: MaterialValueMap
    def __init__(self) -> None:
        ...
    @property
    def etchStopDepth(self) -> float:
        ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def etchantFlux(self) -> float:
        ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ionFlux(self) -> float:
        ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def oxygenFlux(self) -> float:
        ...
    @oxygenFlux.setter
    def oxygenFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def polymerFlux(self) -> float:
        ...
    @polymerFlux.setter
    def polymerFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CF4O2ParametersIons:
    def __init__(self) -> None:
        ...
    @property
    def exponent(self) -> float:
        ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inflectAngle(self) -> float:
        ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meanEnergy(self) -> float:
        ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def minAngle(self) -> float:
        ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_l(self) -> float:
        ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigmaEnergy(self) -> float:
        ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CF4O2ParametersMask:
    def __init__(self) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CF4O2ParametersPassivation:
    def __init__(self) -> None:
        ...
    @property
    def A_C_ie(self) -> float:
        ...
    @A_C_ie.setter
    def A_C_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def A_O_ie(self) -> float:
        ...
    @A_O_ie.setter
    def A_O_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_C_ie(self) -> float:
        ...
    @Eth_C_ie.setter
    def Eth_C_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_O_ie(self) -> float:
        ...
    @Eth_O_ie.setter
    def Eth_O_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CF4O2ParametersSi:
    def __init__(self) -> None:
        ...
    @property
    def A_ie(self) -> float:
        ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_ie(self) -> float:
        ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta_sigma(self) -> float:
        ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_sigma(self) -> float:
        ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CF4O2ParametersSiGe:
    def __init__(self) -> None:
        ...
    def k_sigma_SiGe(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    @property
    def A_ie(self) -> float:
        ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_ie(self) -> float:
        ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta_sigma(self) -> float:
        ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_sigma(self) -> float:
        ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CoverageParameters:
    initialized: bool
    def __init__(self) -> None:
        ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the coverage parameters to a metadata dict.
        """
    def toMetaDataString(self) -> str:
        """
        Convert the coverage parameters to a metadata string.
        """
    @property
    def maxIterations(self) -> int:
        ...
    @maxIterations.setter
    def maxIterations(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tolerance(self) -> float:
        ...
    @tolerance.setter
    def tolerance(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Extrude:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, inputDomain: viennaps.d2.Domain, outputDomain: viennaps.d3.Domain, extent: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"], extrusionAxis: typing.SupportsInt | typing.SupportsIndex, boundaryConditions: typing.Annotated[collections.abc.Sequence[viennals._core.BoundaryConditionEnum], "FixedSize(3)"]) -> None:
        ...
    def apply(self) -> None:
        """
        Run the extrusion.
        """
    def setBoundaryConditions(self, arg0: typing.Annotated[collections.abc.Sequence[viennals._core.BoundaryConditionEnum], "FixedSize(3)"]) -> None:
        """
        Set the boundary conditions in the extruded domain.
        """
    def setExtent(self, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"]) -> None:
        """
        Set the min and max extent in the extruded dimension.
        """
    def setExtrusionAxis(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
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
    def __init__(self) -> None:
        ...
    @property
    def cageAngle(self) -> float:
        ...
    @cageAngle.setter
    def cageAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FluorocarbonMaterialParameters:
    id: Material
    def __init__(self) -> None:
        ...
    @property
    def A_ie(self) -> float:
        ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def B_sp(self) -> float:
        ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def E_a(self) -> float:
        ...
    @E_a.setter
    def E_a(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_ie(self) -> float:
        ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def K(self) -> float:
        ...
    @K.setter
    def K(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta_e(self) -> float:
        ...
    @beta_e.setter
    def beta_e(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta_p(self) -> float:
        ...
    @beta_p.setter
    def beta_p(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def density(self) -> float:
        ...
    @density.setter
    def density(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FluorocarbonParameters:
    Ions: FluorocarbonParametersIons
    def __init__(self) -> None:
        ...
    def addMaterial(self, materialParameters: FluorocarbonMaterialParameters) -> None:
        ...
    def getMaterialParameters(self, material: Material) -> FluorocarbonMaterialParameters:
        ...
    @property
    def delta_p(self) -> float:
        ...
    @delta_p.setter
    def delta_p(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def etchStopDepth(self) -> float:
        ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def etchantFlux(self) -> float:
        ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ionFlux(self) -> float:
        ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_ev(self) -> float:
        ...
    @k_ev.setter
    def k_ev(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_ie(self) -> float:
        ...
    @k_ie.setter
    def k_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def polyFlux(self) -> float:
        ...
    @polyFlux.setter
    def polyFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def temperature(self) -> float:
        ...
    @temperature.setter
    def temperature(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FluorocarbonParametersIons:
    def __init__(self) -> None:
        ...
    @property
    def exponent(self) -> float:
        ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inflectAngle(self) -> float:
        ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meanEnergy(self) -> float:
        ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def minAngle(self) -> float:
        ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_l(self) -> float:
        ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigmaEnergy(self) -> float:
        ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FluxEngineType(enum.IntEnum):
    AUTO: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.AUTO: 0>
    CPU_DISK: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.CPU_DISK: 1>
    CPU_TRIANGLE: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.CPU_TRIANGLE: 2>
    GPU_DISK: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.GPU_DISK: 3>
    GPU_LINE: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.GPU_LINE: 5>
    GPU_TRIANGLE: typing.ClassVar[FluxEngineType]  # value = <FluxEngineType.GPU_TRIANGLE: 4>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class HoleShape(enum.IntEnum):
    FULL: typing.ClassVar[HoleShape]  # value = <HoleShape.FULL: 0>
    HALF: typing.ClassVar[HoleShape]  # value = <HoleShape.HALF: 1>
    QUARTER: typing.ClassVar[HoleShape]  # value = <HoleShape.QUARTER: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class IBEParameters:
    cos4Yield: IBEParametersCos4Yield
    materialPlaneWaferRate: MaterialValueMap
    def __init__(self) -> None:
        ...
    def toProcessMetaData(self) -> dict[str, list[float]]:
        """
        Convert the IBE parameters to a metadata dict.
        """
    @property
    def exponent(self) -> float:
        ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inflectAngle(self) -> float:
        ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meanEnergy(self) -> float:
        ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def minAngle(self) -> float:
        ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_l(self) -> float:
        ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def planeWaferRate(self) -> float:
        ...
    @planeWaferRate.setter
    def planeWaferRate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def redepositionRate(self) -> float:
        ...
    @redepositionRate.setter
    def redepositionRate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def redepositionThreshold(self) -> float:
        ...
    @redepositionThreshold.setter
    def redepositionThreshold(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigmaEnergy(self) -> float:
        ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thetaRMax(self) -> float:
        ...
    @thetaRMax.setter
    def thetaRMax(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thetaRMin(self) -> float:
        ...
    @thetaRMin.setter
    def thetaRMin(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thresholdEnergy(self) -> float:
        ...
    @thresholdEnergy.setter
    def thresholdEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def tiltAngle(self) -> float:
        ...
    @tiltAngle.setter
    def tiltAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class IBEParametersCos4Yield:
    isDefined: bool
    def __init__(self) -> None:
        ...
    def aSum(self) -> float:
        ...
    @property
    def a1(self) -> float:
        ...
    @a1.setter
    def a1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a2(self) -> float:
        ...
    @a2.setter
    def a2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a3(self) -> float:
        ...
    @a3.setter
    def a3(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a4(self) -> float:
        ...
    @a4.setter
    def a4(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class Length:
    @staticmethod
    def convertAngstrom() -> float:
        ...
    @staticmethod
    def convertCentimeter() -> float:
        ...
    @staticmethod
    def convertMeter() -> float:
        ...
    @staticmethod
    def convertMicrometer() -> float:
        ...
    @staticmethod
    def convertMillimeter() -> float:
        ...
    @staticmethod
    def convertNanometer() -> float:
        ...
    @staticmethod
    def getInstance() -> Length:
        ...
    @staticmethod
    def setUnit(arg0: str) -> None:
        ...
    @staticmethod
    def toShortString() -> str:
        ...
    @staticmethod
    def toString() -> str:
        ...
class LengthUnit(enum.IntEnum):
    ANGSTROM: typing.ClassVar[LengthUnit]  # value = <LengthUnit.ANGSTROM: 5>
    CENTIMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.CENTIMETER: 1>
    METER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.METER: 0>
    MICROMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.MICROMETER: 3>
    MILLIMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.MILLIMETER: 2>
    NANOMETER: typing.ClassVar[LengthUnit]  # value = <LengthUnit.NANOMETER: 4>
    UNDEFINED: typing.ClassVar[LengthUnit]  # value = <LengthUnit.UNDEFINED: 6>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class Logger:
    @staticmethod
    def appendToLogFile(arg0: str) -> bool:
        ...
    @staticmethod
    def closeLogFile() -> None:
        ...
    @staticmethod
    def getInstance() -> Logger:
        ...
    @staticmethod
    def getLogLevel() -> int:
        ...
    @staticmethod
    def setLogFile(arg0: str) -> bool:
        ...
    @staticmethod
    def setLogLevel(arg0: ...) -> None:
        ...
    def addDebug(self, arg0: str) -> Logger:
        ...
    def addError(self, s: str, shouldAbort: bool = True) -> Logger:
        ...
    def addInfo(self, arg0: str) -> Logger:
        ...
    @typing.overload
    def addTiming(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex) -> Logger:
        ...
    @typing.overload
    def addTiming(self, arg0: str, arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex) -> Logger:
        ...
    def addWarning(self, arg0: str) -> Logger:
        ...
    def print(self) -> None:
        ...
class Material:
    ARC: typing.ClassVar[Material]  # value = Material('ARC')
    AZO: typing.ClassVar[Material]  # value = Material('AZO')
    Air: typing.ClassVar[Material]  # value = Material('Air')
    Al2O3: typing.ClassVar[Material]  # value = Material('Al2O3')
    AlN: typing.ClassVar[Material]  # value = Material('AlN')
    Au: typing.ClassVar[Material]  # value = Material('Au')
    BN: typing.ClassVar[Material]  # value = Material('BN')
    BPSG: typing.ClassVar[Material]  # value = Material('BPSG')
    BulkSi: typing.ClassVar[Material]  # value = Material('BulkSi')
    C: typing.ClassVar[Material]  # value = Material('C')
    Co: typing.ClassVar[Material]  # value = Material('Co')
    CoW: typing.ClassVar[Material]  # value = Material('CoW')
    Cr: typing.ClassVar[Material]  # value = Material('Cr')
    Cu: typing.ClassVar[Material]  # value = Material('Cu')
    Custom: typing.ClassVar[Material]  # value = Material('Custom')
    Dielectric: typing.ClassVar[Material]  # value = Material('Dielectric')
    GAS: typing.ClassVar[Material]  # value = Material('GAS')
    GST: typing.ClassVar[Material]  # value = Material('GST')
    GaAs: typing.ClassVar[Material]  # value = Material('GaAs')
    GaN: typing.ClassVar[Material]  # value = Material('GaN')
    Ge: typing.ClassVar[Material]  # value = Material('Ge')
    Graphene: typing.ClassVar[Material]  # value = Material('Graphene')
    HSQ: typing.ClassVar[Material]  # value = Material('HSQ')
    HfO2: typing.ClassVar[Material]  # value = Material('HfO2')
    ITO: typing.ClassVar[Material]  # value = Material('ITO')
    InGaAs: typing.ClassVar[Material]  # value = Material('InGaAs')
    InP: typing.ClassVar[Material]  # value = Material('InP')
    Ir: typing.ClassVar[Material]  # value = Material('Ir')
    La2O3: typing.ClassVar[Material]  # value = Material('La2O3')
    Mask: typing.ClassVar[Material]  # value = Material('Mask')
    Metal: typing.ClassVar[Material]  # value = Material('Metal')
    Mn: typing.ClassVar[Material]  # value = Material('Mn')
    MnN: typing.ClassVar[Material]  # value = Material('MnN')
    MnO: typing.ClassVar[Material]  # value = Material('MnO')
    Mo: typing.ClassVar[Material]  # value = Material('Mo')
    MoS2: typing.ClassVar[Material]  # value = Material('MoS2')
    MoSi2: typing.ClassVar[Material]  # value = Material('MoSi2')
    Ni: typing.ClassVar[Material]  # value = Material('Ni')
    NiW: typing.ClassVar[Material]  # value = Material('NiW')
    PHS: typing.ClassVar[Material]  # value = Material('PHS')
    PMMA: typing.ClassVar[Material]  # value = Material('PMMA')
    PSG: typing.ClassVar[Material]  # value = Material('PSG')
    Pd: typing.ClassVar[Material]  # value = Material('Pd')
    PolySi: typing.ClassVar[Material]  # value = Material('PolySi')
    Polymer: typing.ClassVar[Material]  # value = Material('Polymer')
    Pt: typing.ClassVar[Material]  # value = Material('Pt')
    Rh: typing.ClassVar[Material]  # value = Material('Rh')
    Ru: typing.ClassVar[Material]  # value = Material('Ru')
    RuTa: typing.ClassVar[Material]  # value = Material('RuTa')
    SOC: typing.ClassVar[Material]  # value = Material('SOC')
    SOG: typing.ClassVar[Material]  # value = Material('SOG')
    Si: typing.ClassVar[Material]  # value = Material('Si')
    Si3N4: typing.ClassVar[Material]  # value = Material('Si3N4')
    SiBCN: typing.ClassVar[Material]  # value = Material('SiBCN')
    SiC: typing.ClassVar[Material]  # value = Material('SiC')
    SiCN: typing.ClassVar[Material]  # value = Material('SiCN')
    SiCOH: typing.ClassVar[Material]  # value = Material('SiCOH')
    SiC_HM: typing.ClassVar[Material]  # value = Material('SiC_HM')
    SiGaN: typing.ClassVar[Material]  # value = Material('SiGaN')
    SiGe: typing.ClassVar[Material]  # value = Material('SiGe')
    SiLK: typing.ClassVar[Material]  # value = Material('SiLK')
    SiN: typing.ClassVar[Material]  # value = Material('SiN')
    SiN_HM: typing.ClassVar[Material]  # value = Material('SiN_HM')
    SiO2: typing.ClassVar[Material]  # value = Material('SiO2')
    SiO2_HM: typing.ClassVar[Material]  # value = Material('SiO2_HM')
    SiOCH: typing.ClassVar[Material]  # value = Material('SiOCH')
    SiOCN: typing.ClassVar[Material]  # value = Material('SiOCN')
    SiON: typing.ClassVar[Material]  # value = Material('SiON')
    SiON_HM: typing.ClassVar[Material]  # value = Material('SiON_HM')
    Ta: typing.ClassVar[Material]  # value = Material('Ta')
    Ta2O5: typing.ClassVar[Material]  # value = Material('Ta2O5')
    TaN: typing.ClassVar[Material]  # value = Material('TaN')
    Ti: typing.ClassVar[Material]  # value = Material('Ti')
    TiAlN: typing.ClassVar[Material]  # value = Material('TiAlN')
    TiN: typing.ClassVar[Material]  # value = Material('TiN')
    TiO: typing.ClassVar[Material]  # value = Material('TiO')
    TiO2: typing.ClassVar[Material]  # value = Material('TiO2')
    TiSi2: typing.ClassVar[Material]  # value = Material('TiSi2')
    Undefined: typing.ClassVar[Material]  # value = Material('Undefined')
    VO2: typing.ClassVar[Material]  # value = Material('VO2')
    W: typing.ClassVar[Material]  # value = Material('W')
    WS2: typing.ClassVar[Material]  # value = Material('WS2')
    WSe2: typing.ClassVar[Material]  # value = Material('WSe2')
    WSi2: typing.ClassVar[Material]  # value = Material('WSi2')
    Y2O3: typing.ClassVar[Material]  # value = Material('Y2O3')
    ZnO: typing.ClassVar[Material]  # value = Material('ZnO')
    ZrO: typing.ClassVar[Material]  # value = Material('ZrO')
    ZrO2: typing.ClassVar[Material]  # value = Material('ZrO2')
    aC: typing.ClassVar[Material]  # value = Material('aC')
    aSi: typing.ClassVar[Material]  # value = Material('aSi')
    hBN: typing.ClassVar[Material]  # value = Material('hBN')
    @staticmethod
    def custom(id: typing.SupportsInt | typing.SupportsIndex) -> Material:
        ...
    def __eq__(self, arg0: Material) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: BuiltInMaterial) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, arg0: Material) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def builtIn(self) -> BuiltInMaterial:
        ...
    def customId(self) -> int:
        ...
    def isBuiltIn(self) -> bool:
        ...
    def isCustom(self) -> bool:
        ...
    def kind(self) -> ...:
        ...
    def legacyId(self) -> int:
        ...
class MaterialCategory(enum.IntEnum):
    Compound: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Compound: 6>
    Generic: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Generic: 0>
    Hardmask: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Hardmask: 3>
    Metal: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Metal: 4>
    Misc: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Misc: 9>
    OxideNitride: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.OxideNitride: 2>
    Silicide: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Silicide: 5>
    Silicon: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.Silicon: 1>
    TCO: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.TCO: 8>
    TwoD: typing.ClassVar[MaterialCategory]  # value = <MaterialCategory.TwoD: 7>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class MaterialInfo:
    category: ...
    conductive: bool
    name: str
    @property
    def color_hex(self) -> int:
        ...
    @color_hex.setter
    def color_hex(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def color_rgb(self) -> str:
        ...
    @property
    def density_gcm3(self) -> float:
        ...
    @density_gcm3.setter
    def density_gcm3(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MaterialKind(enum.IntEnum):
    BuiltIn: typing.ClassVar[MaterialKind]  # value = <MaterialKind.BuiltIn: 0>
    Custom: typing.ClassVar[MaterialKind]  # value = <MaterialKind.Custom: 1>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class MaterialMap:
    @staticmethod
    def fromString(name: str) -> Material:
        """
        Resolve built-in or register custom material by name.
        """
    @staticmethod
    @typing.overload
    def isMaterial(arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: Material) -> bool:
        ...
    @staticmethod
    @typing.overload
    def isMaterial(*args, **kwargs) -> bool:
        ...
    @staticmethod
    def mapToMaterial(arg0: typing.SupportsFloat | typing.SupportsIndex) -> Material:
        """
        Map a float to a material.
        """
    @staticmethod
    @typing.overload
    def toString(arg0: Material) -> str:
        """
        Get the name of a material.
        """
    @staticmethod
    @typing.overload
    def toString(arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        """
        Get the name of a material.
        """
    def __init__(self) -> None:
        ...
    def getMaterialAtIdx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> Material:
        ...
    def getMaterialIdAtIdx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> int:
        ...
    def getMaterialMap(self) -> viennals._core.MaterialMap:
        ...
    def insertNextMaterial(self, material: Material) -> None:
        ...
    def size(self) -> int:
        ...
class MaterialRegistry:
    @staticmethod
    def instance() -> MaterialRegistry:
        ...
    def customMaterialCount(self) -> int:
        ...
    def findMaterial(self, name: str) -> viennaps._core.Material | None:
        ...
    def getInfo(self, arg0: Material) -> MaterialInfo:
        ...
    def getMaterial(self, name: str) -> Material:
        ...
    def getName(self, material: Material) -> str:
        ...
    @typing.overload
    def hasMaterial(self, arg0: str) -> bool:
        ...
    @typing.overload
    def hasMaterial(self, arg0: Material) -> bool:
        ...
    def isBuiltIn(self, material: Material) -> bool:
        ...
    def registerMaterial(self, name: str) -> Material:
        ...
    def setInfo(self, arg0: Material, arg1: MaterialInfo) -> None:
        ...
class MaterialValueMap:
    def __init__(self) -> None:
        ...
    def clearAll(self) -> None:
        ...
    def get(self, material: Material) -> float:
        ...
    def getDefault(self) -> float:
        ...
    def set(self, material: Material, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def setDefault(self, value: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MetaDataLevel(enum.IntEnum):
    FULL: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.FULL: 3>
    GRID: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.GRID: 1>
    NONE: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.NONE: 0>
    PROCESS: typing.ClassVar[MetaDataLevel]  # value = <MetaDataLevel.PROCESS: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class NormalizationType(enum.IntEnum):
    MAX: typing.ClassVar[NormalizationType]  # value = <NormalizationType.MAX: 1>
    SOURCE: typing.ClassVar[NormalizationType]  # value = <NormalizationType.SOURCE: 0>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class PlasmaEtchingParameters:
    Ions: PlasmaEtchingParametersIons
    Mask: PlasmaEtchingParametersMask
    Passivation: PlasmaEtchingParametersPassivation
    Polymer: PlasmaEtchingParametersPolymer
    Substrate: PlasmaEtchingParametersSubstrate
    beta_E: MaterialValueMap
    beta_P: MaterialValueMap
    rateFactors: MaterialValueMap
    def __init__(self) -> None:
        ...
    @property
    def etchStopDepth(self) -> float:
        ...
    @etchStopDepth.setter
    def etchStopDepth(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def etchantFlux(self) -> float:
        ...
    @etchantFlux.setter
    def etchantFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ionFlux(self) -> float:
        ...
    @ionFlux.setter
    def ionFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def passivationFlux(self) -> float:
        ...
    @passivationFlux.setter
    def passivationFlux(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PlasmaEtchingParametersIons:
    def __init__(self) -> None:
        ...
    @property
    def exponent(self) -> float:
        ...
    @exponent.setter
    def exponent(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def inflectAngle(self) -> float:
        ...
    @inflectAngle.setter
    def inflectAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def meanEnergy(self) -> float:
        ...
    @meanEnergy.setter
    def meanEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def minAngle(self) -> float:
        ...
    @minAngle.setter
    def minAngle(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_l(self) -> float:
        ...
    @n_l.setter
    def n_l(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def sigmaEnergy(self) -> float:
        ...
    @sigmaEnergy.setter
    def sigmaEnergy(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thetaRMax(self) -> float:
        ...
    @thetaRMax.setter
    def thetaRMax(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def thetaRMin(self) -> float:
        ...
    @thetaRMin.setter
    def thetaRMin(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PlasmaEtchingParametersMask:
    def __init__(self) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def B_sp(self) -> float:
        ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PlasmaEtchingParametersPassivation:
    def __init__(self) -> None:
        ...
    @property
    def A_ie(self) -> float:
        ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_ie(self) -> float:
        ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PlasmaEtchingParametersPolymer:
    usePolyCosThetaYield: bool
    def __init__(self) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def B_sp(self) -> float:
        ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a1(self) -> float:
        ...
    @a1.setter
    def a1(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a2(self) -> float:
        ...
    @a2.setter
    def a2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a3(self) -> float:
        ...
    @a3.setter
    def a3(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def a4(self) -> float:
        ...
    @a4.setter
    def a4(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PlasmaEtchingParametersSubstrate:
    def __init__(self) -> None:
        ...
    @property
    def A_ie(self) -> float:
        ...
    @A_ie.setter
    def A_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def A_sp(self) -> float:
        ...
    @A_sp.setter
    def A_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def B_ie(self) -> float:
        ...
    @B_ie.setter
    def B_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def B_sp(self) -> float:
        ...
    @B_sp.setter
    def B_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_ie(self) -> float:
        ...
    @Eth_ie.setter
    def Eth_ie(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def Eth_sp(self) -> float:
        ...
    @Eth_sp.setter
    def Eth_sp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def beta_sigma(self) -> float:
        ...
    @beta_sigma.setter
    def beta_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_sigma(self) -> float:
        ...
    @k_sigma.setter
    def k_sigma(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rho(self) -> float:
        ...
    @rho.setter
    def rho(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class ProcessParams:
    def __init__(self) -> None:
        ...
    @typing.overload
    def getScalarData(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    @typing.overload
    def getScalarData(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
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
    def getScalarDataLabel(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        ...
    def insertNextScalar(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: str) -> None:
        ...
class RateSet:
    calculateVisibility: bool
    def __init__(self, direction: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"] = [0.0, 0.0, 0.0], directionalVelocity: typing.SupportsFloat | typing.SupportsIndex = 0.0, isotropicVelocity: typing.SupportsFloat | typing.SupportsIndex = 0.0, maskMaterials: collections.abc.Sequence[Material] = ..., calculateVisibility: bool = True) -> None:
        ...
    def print(self) -> None:
        ...
    @property
    def direction(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @direction.setter
    def direction(self, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def directionalVelocity(self) -> float:
        ...
    @directionalVelocity.setter
    def directionalVelocity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def isotropicVelocity(self) -> float:
        ...
    @isotropicVelocity.setter
    def isotropicVelocity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def maskMaterials(self) -> list[Material]:
        ...
    @maskMaterials.setter
    def maskMaterials(self, arg0: collections.abc.Sequence[Material]) -> None:
        ...
class RayTracingParameters:
    ignoreFluxBoundaries: bool
    normalizationType: NormalizationType
    useRandomSeeds: bool
    def __init__(self) -> None:
        ...
    def toMetaData(self) -> dict[str, list[float]]:
        """
        Convert the ray tracing parameters to a metadata dict.
        """
    def toMetaDataString(self) -> str:
        """
        Convert the ray tracing parameters to a metadata string.
        """
    @property
    def diskRadius(self) -> float:
        ...
    @diskRadius.setter
    def diskRadius(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def maxBoundaryHits(self) -> int:
        ...
    @maxBoundaryHits.setter
    def maxBoundaryHits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def maxReflections(self) -> int:
        ...
    @maxReflections.setter
    def maxReflections(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def minNodeDistanceFactor(self) -> float:
        ...
    @minNodeDistanceFactor.setter
    def minNodeDistanceFactor(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def raysPerPoint(self) -> int:
        ...
    @raysPerPoint.setter
    def raysPerPoint(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def rngSeed(self) -> int:
        ...
    @rngSeed.setter
    def rngSeed(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def smoothingNeighbors(self) -> int:
        ...
    @smoothingNeighbors.setter
    def smoothingNeighbors(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class RenderMode(enum.IntEnum):
    INTERFACE: typing.ClassVar[RenderMode]  # value = <RenderMode.INTERFACE: 1>
    SURFACE: typing.ClassVar[RenderMode]  # value = <RenderMode.SURFACE: 0>
    VOLUME: typing.ClassVar[RenderMode]  # value = <RenderMode.VOLUME: 2>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class Slice:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, inputDomain: viennaps.d3.Domain, outputDomain: viennaps.d2.Domain, sliceDimension: typing.SupportsInt | typing.SupportsIndex, slicePosition: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
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
    def setSliceDimension(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set the dimension along which to slice (0, 1).
        """
    def setSlicePosition(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set the position along the slice dimension at which to slice.
        """
class Time:
    @staticmethod
    def convertMillisecond() -> float:
        ...
    @staticmethod
    def convertMinute() -> float:
        ...
    @staticmethod
    def convertSecond() -> float:
        ...
    @staticmethod
    def getInstance() -> Time:
        ...
    @staticmethod
    def setUnit(arg0: str) -> None:
        ...
    @staticmethod
    def toShortString() -> str:
        ...
    @staticmethod
    def toString() -> str:
        ...
class TimeUnit(enum.IntEnum):
    MILLISECOND: typing.ClassVar[TimeUnit]  # value = <TimeUnit.MILLISECOND: 2>
    MINUTE: typing.ClassVar[TimeUnit]  # value = <TimeUnit.MINUTE: 0>
    SECOND: typing.ClassVar[TimeUnit]  # value = <TimeUnit.SECOND: 1>
    UNDEFINED: typing.ClassVar[TimeUnit]  # value = <TimeUnit.UNDEFINED: 3>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
def gpuAvailable() -> bool:
    """
    Check if ViennaPS was compiled with GPU support.
    """
def setNumThreads(arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
__version__: str = '4.3.0'
version: str = '4.3.0'
