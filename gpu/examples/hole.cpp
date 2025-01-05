#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>

#include <gpu/vcContext.hpp>

#include <pscuProcess.hpp>
#include <pscuSF6O2Etching.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;

  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;
  CreateContext(context);

  const NumericType gridDelta = 1.9;
  const NumericType extent = 100.;
  const NumericType holeRadius = 15.;
  const NumericType maskHeight = 100.;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeHole<NumericType, D>(domain, gridDelta, extent, extent, holeRadius,
                           maskHeight, 0.f, 0.f, false, true, Material::Si)
      .apply();

  viennaps::SF6O2Parameters<NumericType> params;

  params.oxygenFlux = 5.;
  params.etchantFlux = 2000.;
  params.ionFlux = 15.;

  params.Ions.meanEnergy = 100.;
  params.Ions.sigmaEnergy = 10.;
  params.Ions.exponent = 1000.;

  auto model = SmartPointer<gpu::SF6O2Etching<NumericType, D>>::New(params);

  gpu::Process<NumericType, D> process(context, domain, model);
  process.setNumberOfRaysPerPoint(3000);
  process.setProcessParams(params);
  process.setMaxCoverageInitIterations(10);
  process.setProcessDuration(50.);
  process.apply();

  domain->saveSurfaceMesh("final.vtp");
}