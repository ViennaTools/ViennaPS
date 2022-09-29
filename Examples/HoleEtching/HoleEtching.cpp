#include <SF6O2Etching.hpp>
#include <psMakeHole.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeHole<NumericType, D>(geometry, 0.25 /* grid delta */, 15 /*x extent*/,
                             15 /*y extent*/, 5 /*hole radius*/,
                             1 /* mask height*/, true /*create mask*/)
      .apply();

  SF6O2Etching<NumericType, D> model(
      2e16 /*ion flux*/, 4.5e18 /*etchant flux*/, 1.e18 /*oxygen flux*/,
      100 /*mean ion energy (eV)*/, 0 /*mask material ID*/);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model.getProcessModel());
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(3000);
  process.setProcessDuration(200);

  auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "initial.vtp").apply();

  process.apply();

  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "final.vtp").apply();

  return EXIT_SUCCESS;
}
