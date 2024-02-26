#include <geometries/psMakeFin.hpp>
#include <psPlasmaDamage.hpp>
#include <psProcess.hpp>

#include "parameters.hpp"

int main(int argc, char *argv[]) {
  using NumericType = float;
  constexpr int D = 3;

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

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeFin<NumericType, D>(geometry, params.gridDelta, params.xExtent,
                            params.yExtent, params.finWidth, params.finHeight,
                            0. /*taper angle*/, 0. /* base height*/,
                            false /*periodic boundary*/, false /*create mask*/,
                            psMaterial::Si /*material*/)
      .apply();

  // generate cell set with depth 5
  geometry->generateCellSet(-5. /*depth*/, psMaterial::Si,
                            false /*cell set below surface*/);

  auto model = psSmartPointer<psPlasmaDamage<NumericType, D>>::New(
      params.ionEnergy /* mean ion energy (eV) */,
      params.meanFreePath /* damage ion mean free path */,
      psMaterial::None /*mask material (no mask)*/);

  psProcess<NumericType, D>(geometry, model, 0.)
      .apply(); // apply only damage model

  geometry->getCellSet()->writeVTU("DamageModel.vtu");
}
