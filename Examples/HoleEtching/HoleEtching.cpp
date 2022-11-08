#include <Geometries/psMakeHole.hpp>
#include <SF6O2Etching.hpp>
#include <psProcess.hpp>
#include <psToSurfaceMesh.hpp>
#include <psVTKWriter.hpp>

#include "ConfigParser.hpp"

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

// Parse the parameters
  int P, y;
  NumericType topRadius, maskHeight, taperAngle, processTime, totalEtchantFlux, totalOxygenFlux, totalIonFlux, A_O;
  NumericType bottomRadius = -1;

  if (argc > 1) {
    auto config = parseConfig<NumericType>(argv[1]);
    if (config.size() == 0) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    for (auto [key, value] : config) {
      if (key == "topRadius") {
        topRadius = value;
      } else if (key == "P") {
        P = value;
      } else if (key == "y") {
        y = value;
      } else if (key == "bottomRadius") {
        bottomRadius = value;
      } else if (key == "maskHeight") {
        maskHeight = value;
      } else if (key == "taperAngle") {
        taperAngle = value;
      } else if (key == "processTime") {
        processTime = value;
      } else if (key == "totalEtchantFlux") {
        totalEtchantFlux = value;
      } else if (key == "totalOxygenFlux") {
        totalOxygenFlux = value;
      } else if (key == "totalIonFlux") {
        totalIonFlux = value;
      } else if (key == "A_O") {
        A_O = value;
      }
    }
  }

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeHole<NumericType, D>(geometry, 0.02 /* grid delta */, 1 /*x extent*/,
                             1 /*y extent*/, 0.2 /*hole radius*/,
                             1.2 /* mask height*/, true /*create mask*/)
      .apply();

  SF6O2Etching<NumericType, D> model(
      totalIonFlux /*ion flux*/, totalEtchantFlux /*etchant flux*/, totalOxygenFlux /*oxygen flux*/,
      100 /*min ion energy (eV)*/, 3 /*oxy sputter yield*/, 0 /*mask material ID*/);

  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model.getProcessModel());
  process.setMaxCoverageInitIterations(10);
  process.setNumberOfRaysPerPoint(50);
  process.setProcessDuration(processTime);

  auto mesh = psSmartPointer<lsMesh<NumericType>>::New();
  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "initial.vtp").apply();

  process.apply();

  psToSurfaceMesh<NumericType, D>(geometry, mesh).apply();
  psVTKWriter<NumericType>(mesh, "final.vtp").apply();

  return EXIT_SUCCESS;
}
