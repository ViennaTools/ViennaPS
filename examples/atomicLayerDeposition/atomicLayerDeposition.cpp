#include <psAtomicLayerModel.hpp>
#include <psDomain.hpp>
#include <psMakeTrench.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  constexpr int D = 2;
  using NumericType = double;
  omp_set_num_threads(12);

  // Parse the parameters
  Parameters<NumericType> params;
  if (argc > 1) {
    auto config = psUtils::readConfigFile(argv[1]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  // Create a domain
  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(domain, params.gridDelta, params.xExtent,
                               params.yExtent, params.trenchWidth,
                               params.trenchHeight, params.taperAngle, 0.,
                               false, false, psMaterial::Si)
      .apply();
  domain->generateCellSet(params.topSpace, psMaterial::GAS, true);
  auto cellSet = domain->getCellSet();

  psAtomicLayerModel<NumericType, D> model(domain, params.diffusionCoefficient,
                                           params.inFlux, params.adsorptionRate,
                                           params.depositionThreshold);

  double time = 0.;
  int i = 0;
  while (time < params.processTime) {
    time += model.timeStep(false);
    if (time - i > 0.1) {
      cellSet->writeVTU("ALD_" + std::to_string(i++) + ".vtu");
      std::cout << "Time: " << time << std::endl;
    }
  }
}