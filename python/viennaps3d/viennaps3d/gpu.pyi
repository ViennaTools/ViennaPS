"""
GPU accelerated functions.
"""
from __future__ import annotations
import collections.abc
import typing
import viennals3d.viennals3d
import viennaps3d.viennaps3d
__all__: list[str] = ['Context', 'FaradayCageEtching', 'HBrO2Etching', 'MultiParticleProcess', 'Path', 'Process', 'ProcessModel', 'SF6O2Etching', 'SingleParticleProcess']
class Context:
    def __init__(self) -> None:
        ...
    def addModule(self, arg0: str) -> None:
        """
        Add a module to the context.
        """
    def create(self, modulePath: Path = '', deviceID: typing.SupportsInt = 0) -> None:
        """
        Create a new context.
        """
    def destroy(self) -> None:
        """
        Destroy the context.
        """
    def getModulePath(self) -> str:
        """
        Get the module path.
        """
    @property
    def deviceID(self) -> int:
        """
        Device ID.
        """
    @deviceID.setter
    def deviceID(self, arg0: typing.SupportsInt) -> None:
        ...
class FaradayCageEtching(ProcessModel):
    def __init__(self, rate: typing.SupportsFloat, stickingProbability: typing.SupportsFloat, power: typing.SupportsFloat, cageAngle: typing.SupportsFloat, tiltAngle: typing.SupportsFloat) -> None:
        ...
class HBrO2Etching(ProcessModel):
    def __init__(self, parameters: viennaps3d.viennaps3d.PlasmaEtchingParameters) -> None:
        ...
class MultiParticleProcess(ProcessModel):
    def __init__(self) -> None:
        ...
    def addIonParticle(self, sourcePower: typing.SupportsFloat, thetaRMin: typing.SupportsFloat = 0.0, thetaRMax: typing.SupportsFloat = 90.0, minAngle: typing.SupportsFloat = 0.0, B_sp: typing.SupportsFloat = -1.0, meanEnergy: typing.SupportsFloat = 0.0, sigmaEnergy: typing.SupportsFloat = 0.0, thresholdEnergy: typing.SupportsFloat = 0.0, inflectAngle: typing.SupportsFloat = 0.0, n: typing.SupportsFloat = 1, label: str = 'ionFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self, stickingProbability: typing.SupportsFloat, label: str = 'neutralFlux') -> None:
        ...
    @typing.overload
    def addNeutralParticle(self, materialSticking: collections.abc.Mapping[viennaps3d.viennaps3d.Material, typing.SupportsFloat], defaultStickingProbability: typing.SupportsFloat = 1.0, label: str = 'neutralFlux') -> None:
        ...
    def setRateFunction(self, arg0: collections.abc.Callable[[collections.abc.Sequence[typing.SupportsFloat], viennaps3d.viennaps3d.Material], float]) -> None:
        ...
class Path:
    def __init__(self, arg0: str) -> None:
        ...
class Process:
    @typing.overload
    def __init__(self, context: Context) -> None:
        ...
    @typing.overload
    def __init__(self, context: Context, domain: viennaps3d.viennaps3d.Domain, model: ProcessModel, duration: typing.SupportsFloat) -> None:
        ...
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
    def getAdvectionParameters(self) -> viennaps3d.viennaps3d.AdvectionParameters:
        """
        Get the advection parameters for the process.
        """
    def getProcessDuration(self) -> float:
        """
        Returns the duration of the recently run process. This duration can sometimes slightly vary from the set process duration, due to the maximum time step according to the CFL condition.
        """
    def getRayTracingParameters(self) -> viennaps3d.viennaps3d.RayTracingParameters:
        """
        Get the ray tracing parameters for the process.
        """
    def setAdvectionParameters(self, arg0: viennaps3d.viennaps3d.AdvectionParameters) -> None:
        """
        Set the advection parameters for the process.
        """
    def setCoverageDeltaThreshold(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the threshold for the coverage delta metric to reach convergence.
        """
    def setDomain(self, arg0: viennaps3d.viennaps3d.Domain) -> None:
        """
        Set the process domain.
        """
    def setIntegrationScheme(self, arg0: viennals3d.viennals3d.IntegrationSchemeEnum) -> None:
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
    def setRayTracingParameters(self, arg0: viennaps3d.viennaps3d.RayTracingParameters) -> None:
        """
        Set the ray tracing parameters for the process.
        """
    def setTimeStepRatio(self, arg0: typing.SupportsFloat) -> None:
        """
        Set the CFL condition to use during advection. The CFL condition sets the maximum distance a surface can be moved during one advection step. It MUST be below 0.5 to guarantee numerical stability. Defaults to 0.4999.
        """
class ProcessModel:
    pass
class SF6O2Etching(ProcessModel):
    def __init__(self, parameters: viennaps3d.viennaps3d.PlasmaEtchingParameters) -> None:
        ...
class SingleParticleProcess(ProcessModel):
    @typing.overload
    def __init__(self, rate: typing.SupportsFloat = 1.0, stickingProbability: typing.SupportsFloat = 1.0, sourceExponent: typing.SupportsFloat = 1.0, maskMaterial: viennaps3d.viennaps3d.Material = ...) -> None:
        ...
    @typing.overload
    def __init__(self, rate: typing.SupportsFloat, stickingProbability: typing.SupportsFloat, sourceExponent: typing.SupportsFloat, maskMaterials: collections.abc.Sequence[viennaps3d.viennaps3d.Material]) -> None:
        ...
    @typing.overload
    def __init__(self, materialRates: collections.abc.Mapping[viennaps3d.viennaps3d.Material, typing.SupportsFloat], stickingProbability: typing.SupportsFloat, sourceExponent: typing.SupportsFloat) -> None:
        ...
