#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main() {
  // Dimension of the domain: 2 for Trench, 3 for Hole
  constexpr int D = 2;
  omp_set_num_threads(16);
  using NumericType = double;

  Logger::setLogLevel(LogLevel::INFO);

  // Geometry parameters
  NumericType gridDelta = 0.05;
  NumericType xExtent = 3.0;
  NumericType yExtent = 3.0;
  NumericType featureWidth = 1.0; // Diameter for hole, Width for trench
  NumericType maskHeight = 1.0;
  NumericType taperAngle = 0.0;

  // Single Particle Process Model
  // Parameters: rate, stickingProbability, yield, maskMaterial
  NumericType rate = -2.0;
  NumericType stickingProbability = 0.2;
  NumericType sourcePower = 1.0;
  NumericType processTime = 1.0;

  auto runSimulation = [&](TemporalScheme temporalScheme,
                           bool calcIntermediate) {
    std::string suffix = util::toString(temporalScheme);
    if (calcIntermediate) {
      suffix += "_recalc";
    }
    auto domain =
        SmartPointer<Domain<NumericType, D>>::New(gridDelta, xExtent, yExtent);

    if constexpr (D == 3) {
      // Create a Hole in 3D
      MakeHole<NumericType, D>(domain, featureWidth / 2.0, 0.0, 0.0, maskHeight,
                               taperAngle, HoleShape::QUARTER)
          .apply();
    } else {
      // Create a Trench in 2D
      MakeTrench<NumericType, D>(domain, featureWidth, 0.0, 0.0, maskHeight,
                                 taperAngle, false)
          .apply();
    }

    auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
        rate, stickingProbability, sourcePower, Material::Mask);

    Process<NumericType, D> process(domain, model, processTime);

    const auto fluxEngine = util::convert<FluxEngineType>("CT");

    AdvectionParameters advectionParams;
    advectionParams.spatialScheme = SpatialScheme::WENO_3RD_ORDER;
    advectionParams.temporalScheme = temporalScheme;
    advectionParams.calculateIntermediateVelocities = calcIntermediate;
    process.setParameters(advectionParams);
    process.setFluxEngineType(fluxEngine);

    Logger::getInstance().addInfo("Running simulation: " + suffix).print();

    process.apply();

    domain->saveSurfaceMesh("simpleEtching_" + suffix + ".vtp");
  };

  runSimulation(TemporalScheme::FORWARD_EULER, false);
  runSimulation(TemporalScheme::RUNGE_KUTTA_2ND_ORDER, false);
  runSimulation(TemporalScheme::RUNGE_KUTTA_2ND_ORDER, true);
  runSimulation(TemporalScheme::RUNGE_KUTTA_3RD_ORDER, false);
  runSimulation(TemporalScheme::RUNGE_KUTTA_3RD_ORDER, true);

  return 0;
}
