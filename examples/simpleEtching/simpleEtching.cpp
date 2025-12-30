#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

namespace ps = viennaps;

int main() {
  // Dimension of the domain: 2 for Trench, 3 for Hole
  constexpr int D = 2;
  omp_set_num_threads(16);
  using NumericType = double;

  ps::Logger::setLogLevel(ps::LogLevel::INFO);

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

  auto runSimulation = [&](ps::TemporalScheme temporalScheme,
                           bool calcIntermediate, std::string suffix) {
    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(
        gridDelta, xExtent, yExtent);

    if constexpr (D == 3) {
      // Create a Hole in 3D
      ps::MakeHole<NumericType, D>(domain, featureWidth / 2.0, 0.0, 0.0,
                                   maskHeight, taperAngle,
                                   ps::HoleShape::QUARTER)
          .apply();
    } else {
      // Create a Trench in 2D
      ps::MakeTrench<NumericType, D>(domain, featureWidth, 0.0, 0.0, maskHeight,
                                     taperAngle, false)
          .apply();
    }

    auto model =
        ps::SmartPointer<ps::SingleParticleProcess<NumericType, D>>::New(
            rate, stickingProbability, sourcePower, ps::Material::Mask);

    ps::Process<NumericType, D> process(domain, model, processTime);

    ps::AdvectionParameters advectionParams;
    advectionParams.spatialScheme = ps::SpatialScheme::WENO_5TH_ORDER;
    advectionParams.temporalScheme = temporalScheme;
    advectionParams.calculateIntermediateVelocities = calcIntermediate;
    process.setParameters(advectionParams);

    ps::Logger::getInstance().addInfo("Running simulation: " + suffix).print();

    process.apply();

    domain->saveSurfaceMesh("simpleEtching_" + suffix + ".vtp");
  };

  runSimulation(ps::TemporalScheme::FORWARD_EULER, false, "FE");
  runSimulation(ps::TemporalScheme::RUNGE_KUTTA_2ND_ORDER, false, "RK2");
  runSimulation(ps::TemporalScheme::RUNGE_KUTTA_3RD_ORDER, false, "RK3");
  runSimulation(ps::TemporalScheme::RUNGE_KUTTA_3RD_ORDER, true, "RK3_recalc");

  return 0;
}
