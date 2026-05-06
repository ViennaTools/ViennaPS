#include <geometries/psMakeTrench.hpp>
#include <models/psIonBeamEtching.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 2;

  Logger::setLogLevel(LogLevel::WARNING);

  auto run = [&](FluxEngineType fluxEngine) {
    auto geometry = Domain<NumericType, D>::New(
        0.09, 10.0, BoundaryType::REFLECTIVE_BOUNDARY);
    MakeTrench<NumericType, D>(geometry, 5.0, 0.0, 0.0, 2.0).apply();

    IBEParameters<NumericType> ibeParams;
    ibeParams.exponent = 100;
    ibeParams.thetaRMin = 89.;
    ibeParams.thetaRMax = 90.;

    // ibeParams.meanEnergy = 100;
    // ibeParams.sigmaEnergy = 10;
    // ibeParams.thresholdEnergy = 10;

    ibeParams.planeWaferRate = 1.0;

    // ibeParams.cos4Yield.isDefined = true;
    // ibeParams.cos4Yield.a1 = 1.075;
    // ibeParams.cos4Yield.a2 = -1.55;
    // ibeParams.cos4Yield.a3 = 0.65;

    auto model = SmartPointer<IonBeamEtching<NumericType, D>>::New(
        ibeParams, std::vector<Material>{Material::Mask});

    AdvectionParameters advectionParams;
    advectionParams.spatialScheme =
        viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;

    Process<NumericType, D> process(geometry, model);
    process.setProcessDuration(10);
    process.setParameters(advectionParams);
    process.setFluxEngineType(fluxEngine);

    process.apply();

    geometry->saveSurfaceMesh("ibeEtching_" + util::toString(fluxEngine));
  };

  run(FluxEngineType::GPU_TRIANGLE);
  run(FluxEngineType::GPU_DISK);
  run(FluxEngineType::CPU_TRIANGLE);
  run(FluxEngineType::CPU_DISK);
}
