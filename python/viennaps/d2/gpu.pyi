"""
GPU accelerated functions.
"""

from __future__ import annotations
import collections.abc
import typing
import viennaps._core
import viennaps.d2

__all__: list[str] = [
    "FaradayCageEtching",
    "HBrO2Etching",
    "IonBeamEtching",
    "MultiParticleProcess",
    "ProcessModelGPU",
    "SF6O2Etching",
    "SingleParticleProcess",
]

class FaradayCageEtching(viennaps.d2.ProcessModel):
    def __init__(
        self,
        parameters: viennaps._core.FaradayCageParameters,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...

class HBrO2Etching(ProcessModelGPU):
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None: ...

class IonBeamEtching(viennaps.d2.ProcessModel):
    def __init__(
        self,
        parameters: viennaps._core.IBEParameters,
        maskMaterials: collections.abc.Sequence[viennaps._core.Material],
    ) -> None: ...

class MultiParticleProcess(ProcessModelGPU):
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

class ProcessModelGPU(viennaps.d2.ProcessModelBase):
    pass

class SF6O2Etching(ProcessModelGPU):
    def __init__(self, parameters: viennaps._core.PlasmaEtchingParameters) -> None: ...

class SingleParticleProcess(ProcessModelGPU):
    def __init__(
        self,
        materialRates: collections.abc.Mapping[
            viennaps._core.Material, typing.SupportsFloat
        ],
        rate: typing.SupportsFloat,
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
    ) -> None: ...
