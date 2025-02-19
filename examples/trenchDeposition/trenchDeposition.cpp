#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleProcess.hpp>

#include <psProcess.hpp>
#include <psUtils.hpp>

namespace ps = viennaps;

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  // Parse the parameters
  ps::utils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  ps::MakeTrench<NumericType, D>(
      geometry, params.get("gridDelta") /* grid delta */,
      params.get("xExtent") /*x extent*/, params.get("yExtent") /*y extent*/,
      params.get("trenchWidth") /*trench width*/,
      params.get("trenchHeight") /*trench height*/,
      params.get("taperAngle") /* tapering angle */)
      .apply();

  // copy top layer to capture deposition
  geometry->duplicateTopLevelSet(ps::Material::SiO2);

  auto model = ps::SmartPointer<ps::SingleParticleProcess<NumericType, D>>::New(
      params.get("rate") /*deposition rate*/,
      params.get("stickingProbability") /*particle sticking probability*/,
      params.get("sourcePower") /*particle source power*/);

  ps::Process<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setNumberOfRaysPerPoint(1000);
  process.setProcessDuration(params.get("processTime"));

  geometry->saveHullMesh("initial");

  process.apply();

  geometry->saveHullMesh("final");
}
