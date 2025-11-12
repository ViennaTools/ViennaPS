#include <geometries/psMakePlane.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psSF6O2Etching.hpp>
#include <process/psProcess.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

template <class NumericType, int D> void RunTest() {

  auto domain = Domain<NumericType, D>::New();
  auto model = SmartPointer<ProcessModelCPU<NumericType, D>>::New();

  // constructors
  { Process<NumericType, D> process; }
  { Process<NumericType, D> process(domain); }
  { Process<NumericType, D> process(domain, model, 0.); }

  {
    Logger::setLogLevel(LogLevel::WARNING);
    domain = Domain<NumericType, D>::New(1.0, 50.0, 50.0,
                                         BoundaryType::REFLECTIVE_BOUNDARY);
    MakePlane<NumericType, D>(domain, 0.0).apply();
    model = SmartPointer<IsotropicProcess<NumericType, D>>::New(1.0);
    Process<NumericType, D> process(domain, model, 1.0);

    process.apply();

    MakePlane<NumericType, D>(domain, 10.0, Material::Polymer, true).apply();

    process.apply();
  }

  {
    Logger::setLogLevel(LogLevel::WARNING);
    units::Length::setUnit(units::Length::MICROMETER);
    units::Time::setUnit(units::Time::SECOND);
    domain = Domain<NumericType, D>::New(1.0, 10.0, 10.0,
                                         BoundaryType::REFLECTIVE_BOUNDARY);
    MakePlane<NumericType, D>(domain, 0.0).apply();
    model = SmartPointer<SF6O2Etching<NumericType, D>>::New(
        SF6O2Etching<NumericType, D>::defaultParameters());
    Process<NumericType, D> process(domain, model, 1.0);
    process.setFluxEngineType(FluxEngineType::CPU_DISK);
    CoverageParameters coverageParams;
    coverageParams.maxIterations = 1;
    process.setParameters(coverageParams);

    process.apply();

    MakePlane<NumericType, D>(domain, 10.0, Material::Polymer, true).apply();

    process.apply();
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }