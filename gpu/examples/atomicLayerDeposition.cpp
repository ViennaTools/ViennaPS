#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleALD.hpp>
#include <process/psProcess.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;
  ps::Logger::setLogLevel(ps::LogLevel::TIMING);

  ps::DeviceContext::createContext();

  const NumericType gridDelta = 0.5;
  const NumericType xExtent = 30.0;
  const NumericType yExtent = 10.0;

  const NumericType trenchWidth = 9.0;
  const NumericType trenchDepth = 500.0;

  const int numCycles = 4;
  const NumericType growthPerCycle = 0.2;
  const int totalCycles = 10;

  const NumericType stickingProbability = 4e-3;
  const NumericType evFlux = 0.0;
  const NumericType inFlux = 1000.0;
  const NumericType s0 = 0.336;

  const NumericType pulseTime = 1.0;
  const NumericType coverageTimeStep = 0.05;

  auto geometry = ps::Domain<NumericType, D>::New(gridDelta, xExtent, yExtent);
  ps::MakeTrench<NumericType, D>(geometry, trenchWidth, trenchDepth).apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::Al2O3);

  auto model = ps::SmartPointer<ps::SingleParticleALD<NumericType, D>>::New(
      stickingProbability, numCycles, growthPerCycle, totalCycles,
      coverageTimeStep, evFlux, inFlux, s0, 0.0);

  ps::AtomicLayerProcessParameters alpParams;
  alpParams.numCycles = numCycles;
  alpParams.pulseTime = pulseTime;
  alpParams.coverageTimeStep = coverageTimeStep;
  alpParams.purgePulseTime = 0.0;

  ps::Process<NumericType, D> process(geometry, model);
  process.setFluxEngineType(ps::FluxEngineType::GPU_TRIANGLE);
  process.setAtomicLayerProcessParameters(alpParams);

  geometry->saveSurfaceMesh("initial");

  process.apply();

  geometry->saveSurfaceMesh("final");
}
