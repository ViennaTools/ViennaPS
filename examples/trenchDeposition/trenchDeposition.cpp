#include <psIsotropicProcess.hpp>
#include <psMakeTrench.hpp>
#include <psProcess.hpp>

int main() {
  using NumericType = double;
  constexpr int D = 2;

  auto domain = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(domain, 0.1 /*gridDelta*/, 20. /*xExtent*/,
                               20. /*yExtent*/, 10. /*trenchWidth*/,
                               10. /*trenchDepth*/, 0., 0., false, false,
                               psMaterial::Si)
      .apply();
  // duplicate top layer to capture deposition
  domain->duplicateTopLevelSet(psMaterial::SiO2);

  auto model = psSmartPointer<psIsotropicProcess<NumericType, D>>::New(
      0.1 /*rate*/, psMaterial::None);

  domain->saveVolumeMesh("trench_initial");
  psProcess<NumericType, D>(domain, model, 20.).apply(); // run process for 20s
  domain->saveVolumeMesh("trench_final");
}

// int main(int argc, char *argv[]) {
//   using NumericType = double;
//   constexpr int D = 2;

//   // Parse the parameters
//   Parameters<NumericType> params;
//   if (argc > 1) {
//     auto config = psUtils::readConfigFile(argv[1]);
//     if (config.empty()) {
//       std::cerr << "Empty config provided" << std::endl;
//       return -1;
//     }
//     params.fromMap(config);
//   }

//   auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
//   psMakeTrench<NumericType, D>(
//       geometry, params.gridDelta /* grid delta */, params.xExtent /*x
//       extent*/, params.yExtent /*y extent*/, params.trenchWidth /*trench
//       width*/, params.trenchHeight /*trench height*/, params.taperAngle /*
//       tapering angle */) .apply();

//   // copy top layer to capture deposition
//   geometry->duplicateTopLevelSet();

//   auto model = psSmartPointer<psSingleParticleProcess<NumericType, D>>::New(
//       params.rate /*deposition rate*/,
//       params.stickingProbability /*particle sticking probability*/,
//       params.sourcePower /*particle source power*/);

//   psProcess<NumericType, D> process;
//   process.setDomain(geometry);
//   process.setProcessModel(model);
//   process.setNumberOfRaysPerPoint(1000);
//   process.setProcessDuration(params.processTime);

//   geometry->saveSurfaceMesh("initial.vtp");

//   process.apply();

//   geometry->saveSurfaceMesh("final.vtp");

//   if constexpr (D == 2)
//     geometry->saveVolumeMesh("final");
// }
