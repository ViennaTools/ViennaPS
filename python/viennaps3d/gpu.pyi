import collections.abc
import typing
import viennals3d.viennals3d
import viennaps3d.viennaps3d
from typing import overload

class Context:
    deviceID: int
    def __init__(self) -> None: ...
    def addModule(self, arg0: str) -> None: ...
    def create(
        self, modulePath: Path = ..., deviceID: typing.SupportsInt = ...
    ) -> None: ...
    def destroy(self) -> None: ...
    def getModulePath(self) -> str: ...

class FaradayCageEtching(ProcessModel):
    def __init__(
        self,
        rate: typing.SupportsFloat,
        stickingProbability: typing.SupportsFloat,
        power: typing.SupportsFloat,
        cageAngle: typing.SupportsFloat,
        tiltAngle: typing.SupportsFloat,
    ) -> None: ...

class HBrO2Etching(ProcessModel):
    def __init__(
        self, parameters: viennaps3d.viennaps3d.PlasmaEtchingParameters
    ) -> None: ...

class MultiParticleProcess(ProcessModel):
    def __init__(self) -> None: ...
    def addIonParticle(
        self,
        sourcePower: typing.SupportsFloat,
        thetaRMin: typing.SupportsFloat = ...,
        thetaRMax: typing.SupportsFloat = ...,
        minAngle: typing.SupportsFloat = ...,
        B_sp: typing.SupportsFloat = ...,
        meanEnergy: typing.SupportsFloat = ...,
        sigmaEnergy: typing.SupportsFloat = ...,
        thresholdEnergy: typing.SupportsFloat = ...,
        inflectAngle: typing.SupportsFloat = ...,
        n: typing.SupportsFloat = ...,
        label: str = ...,
    ) -> None: ...
    @overload
    def addNeutralParticle(
        self, stickingProbability: typing.SupportsFloat, label: str = ...
    ) -> None: ...
    @overload
    def addNeutralParticle(
        self,
        materialSticking: collections.abc.Mapping[
            viennaps3d.viennaps3d.Material, typing.SupportsFloat
        ],
        defaultStickingProbability: typing.SupportsFloat = ...,
        label: str = ...,
    ) -> None: ...
    def setRateFunction(
        self,
        arg0: collections.abc.Callable[
            [
                collections.abc.Sequence[typing.SupportsFloat],
                viennaps3d.viennaps3d.Material,
            ],
            float,
        ],
    ) -> None: ...

class Path:
    def __init__(self, arg0: str) -> None: ...

class Process:
    @overload
    def __init__(self, context: Context) -> None: ...
    @overload
    def __init__(
        self,
        context: Context,
        domain: viennaps3d.viennaps3d.Domain,
        model: ProcessModel,
        duration: typing.SupportsFloat,
    ) -> None: ...
    def apply(self) -> None: ...
    def calculateFlux(self) -> viennals3d.viennals3d.Mesh: ...
    def disableAdvectionVelocityOutput(self) -> None: ...
    def disableFluxSmoothing(self) -> None: ...
    def disableRandomSeeds(self) -> None: ...
    def enableAdvectionVelocityOutput(self) -> None: ...
    def enableFluxSmoothing(self) -> None: ...
    def enableRandomSeeds(self) -> None: ...
    def getAdvectionParameters(self) -> viennaps3d.viennaps3d.AdvectionParameters: ...
    def getProcessDuration(self) -> float: ...
    def getRayTracingParameters(self) -> viennaps3d.viennaps3d.RayTracingParameters: ...
    def setAdvectionParameters(
        self, arg0: viennaps3d.viennaps3d.AdvectionParameters
    ) -> None: ...
    def setCoverageDeltaThreshold(self, arg0: typing.SupportsFloat) -> None: ...
    def setDomain(self, arg0: viennaps3d.viennaps3d.Domain) -> None: ...
    def setIntegrationScheme(
        self, arg0: viennals3d.viennals3d.IntegrationSchemeEnum
    ) -> None: ...
    def setMaxCoverageInitIterations(self, arg0: typing.SupportsInt) -> None: ...
    def setNumberOfRaysPerPoint(self, arg0: typing.SupportsInt) -> None: ...
    def setProcessDuration(self, arg0: typing.SupportsFloat) -> None: ...
    def setProcessModel(self, arg0: ProcessModel) -> None: ...
    def setRayTracingParameters(
        self, arg0: viennaps3d.viennaps3d.RayTracingParameters
    ) -> None: ...
    def setTimeStepRatio(self, arg0: typing.SupportsFloat) -> None: ...

class ProcessModel:
    def __init__(self, *args, **kwargs) -> None: ...

class SF6O2Etching(ProcessModel):
    def __init__(
        self, parameters: viennaps3d.viennaps3d.PlasmaEtchingParameters
    ) -> None: ...

class SingleParticleProcess(ProcessModel):
    @overload
    def __init__(
        self,
        rate: typing.SupportsFloat = ...,
        stickingProbability: typing.SupportsFloat = ...,
        sourceExponent: typing.SupportsFloat = ...,
        maskMaterial: viennaps3d.viennaps3d.Material = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        rate: typing.SupportsFloat,
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
        maskMaterials: collections.abc.Sequence[viennaps3d.viennaps3d.Material],
    ) -> None: ...
    @overload
    def __init__(
        self,
        materialRates: collections.abc.Mapping[
            viennaps3d.viennaps3d.Material, typing.SupportsFloat
        ],
        stickingProbability: typing.SupportsFloat,
        sourceExponent: typing.SupportsFloat,
    ) -> None: ...
