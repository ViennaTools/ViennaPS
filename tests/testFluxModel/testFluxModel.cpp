#include <geometries/psMakeHole.hpp>
#include <models/psTestFluxModel.hpp>
#include <process/psProcess.hpp>

int main() {
  using NumericType = double;
  constexpr int D = 3;

  // Create domain
  auto domain = viennaps::Domain<NumericType, D>::New(1.0, 20., 20.);
  viennaps::MakeHole<NumericType, D>(domain, 8.0, 10.0).apply();

  // Create process model
  auto processModel =
      viennaps::SmartPointer<viennaps::TestFluxModel<NumericType, D>>::New(0.3,
                                                                           1.0);

  // Create process
  viennaps::Process<NumericType, D> process(domain, processModel, 1.0);
  process.setFluxEngineType(viennaps::FluxEngineType::GPU_TRIANGLE);

  // Execute process
  auto mesh = process.calculateFlux();

  return 0;
}