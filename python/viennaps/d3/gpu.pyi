"""
GPU accelerated functions.
"""
from __future__ import annotations
import collections.abc
import typing
import viennaps._core
import viennaps.d3
__all__: list[str] = ['FaradayCageEtching', 'HBrO2Etching', 'IonBeamEtching', 'MultiParticleProcess', 'ProcessModelGPU', 'SF6O2Etching', 'SingleParticleProcess']
class FaradayCageEtching(viennaps.d3.ProcessModel):
    def __init__(self, parameters: viennaps._core.FaradayCageParameters, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
class HBrO2Etching(ProcessModelGPU):
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
class IonBeamEtching(viennaps.d3.ProcessModel):
    def __init__(self, parameters: viennaps._core.IBEParameters, maskMaterials: collections.abc.Sequence[viennaps._core.Material]) -> None:
        ...
class MultiParticleProcess(ProcessModelGPU):
    def __init__(self) -> None:
        ...
    def addIonParticle(self, sourcePower: typing.SupportsFloat | typing.SupportsIndex, thetaRMin: typing.SupportsFloat | typing.SupportsIndex = 0.0, thetaRMax: typing.SupportsFloat | typing.SupportsIndex = 90.0, minAngle: typing.SupportsFloat | typing.SupportsIndex = 0.0, B_sp: typing.SupportsFloat | typing.SupportsIndex = -1.0, meanEnergy: typing.SupportsFloat | typing.SupportsIndex = 0.0, sigmaEnergy: typing.SupportsFloat | typing.SupportsIndex = 0.0, thresholdEnergy: typing.SupportsFloat | typing.SupportsIndex = 0.0, inflectAngle: typing.SupportsFloat | typing.SupportsIndex = 0.0, n: typing.SupportsFloat | typing.SupportsIndex = 1, label: str = 'ionFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self, stickingProbability: typing.SupportsFloat | typing.SupportsIndex, label: str = 'neutralFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self, materialSticking: collections.abc.Mapping[viennaps._core.Material, typing.SupportsFloat | typing.SupportsIndex], defaultStickingProbability: typing.SupportsFloat | typing.SupportsIndex = 1.0, label: str = 'neutralFlux') -> None:
        ...
    def setRateFunction(self, arg0: collections.abc.Callable[[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], viennaps._core.Material], float]) -> None:
        ...
class ProcessModelGPU(viennaps.d3.ProcessModelBase):
    pass
class SF6O2Etching(ProcessModelGPU):
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None:
        ...
class SingleParticleProcess(ProcessModelGPU):
    def __init__(self, materialRates: collections.abc.Mapping[viennaps._core.Material, typing.SupportsFloat | typing.SupportsIndex], rate: typing.SupportsFloat | typing.SupportsIndex, stickingProbability: typing.SupportsFloat | typing.SupportsIndex, sourceExponent: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
