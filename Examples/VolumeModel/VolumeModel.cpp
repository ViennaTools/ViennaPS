#include <PlasmaDamage.hpp>
#include <psMakeFin.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

int main(int argc, char *argv[]) {
  using NumericType = float;
  constexpr int D = 3;

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeFin<NumericType, D>(geometry, .25 /*grid delta*/, 10 /*x extent*/,
                            7 /*y extent*/, 5 /*fin width*/, 15 /*fin height*/,
                            false /*create mask*/)
      .apply();
  // generate cell set with depth 5 below the lowest point of the surface
  geometry->generateCellSet(5. /*depth*/, false /*cell set below surface*/);

  PlasmaDamage<NumericType, D> model(100 /*mean ion energy (eV)*/,
                                     0.75 /* damage ion mean free path */,
                                     -1 /*mask material ID (no mask)*/);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model.getProcessModel());
  process.setProcessDuration(0); // apply only damage model

  process.apply();

  geometry->getCellSet()->writeVTU("DamageModel.vtu");

  return EXIT_SUCCESS;
}
