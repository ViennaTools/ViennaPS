#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleALD.hpp>
#include <psAtomicLayerProcess.hpp>
#include <psConstants.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>
#include <psToDiskMesh.hpp>

// #include "constants.hpp"
#include "geometry.hpp"

namespace ps = viennaps;

template <typename T, int D> class MeasureProfile {
  using ResultType = std::pair<std::vector<T>, std::vector<T>>;

  ps::SmartPointer<ps::Domain<T, D>> domain_;
  T cutoffHeight_;

public:
  MeasureProfile(ps::SmartPointer<ps::Domain<T, D>> &domain, T cutoffHeight)
      : cutoffHeight_(cutoffHeight) {
    domain_ = ps::SmartPointer<ps::Domain<T, D>>::New();
    domain_->deepCopy(domain);
  }

  ResultType get() {
    ps::Planarize<T, D>(domain_, cutoffHeight_).apply();
    auto mesh = viennals::Mesh<T>::New();
    ps::ToDiskMesh<T, D>(domain_, mesh).apply();

    std::vector<T> height, position;
    height.reserve(mesh->nodes.size());
    position.reserve(mesh->nodes.size());
    for (const auto &node : mesh->nodes) {
      position.push_back(node[0]);
      height.push_back(node[1]);
    }

    return {position, height};
  }

  void save(const std::string &filename) {
    auto result = get();
    std::ofstream file(filename);
    for (size_t i = 0; i < result.first.size(); ++i) {
      file << result.first[i] << " " << result.second[i] << "\n";
    }
    file.close();
  }
};

int main(int argc, char **argv) {
  constexpr int D = 2;
  using NumericType = double;

  omp_set_num_threads(16);
#ifndef NDEBUG
  omp_set_num_threads(1);
#endif

  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return 1;
  }

  ps::Logger::setLogLevel(ps::LogLevel::DEBUG);

  auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  makeT(domain, params.get("gridDelta"), params.get("openingDepth"),
        params.get("openingWidth"), params.get("gapLength"),
        params.get("gapHeight"), params.get("xPad"), ps::Material::Si,
        params.get("gapWidth"));

  domain->saveVolumeMesh("SingleParticleALD_initial");

  domain->duplicateTopLevelSet(ps::Material::Al2O3);

  auto gasMFP = ps::constants::gasMeanFreePath(params.get("pressure"),
                                               params.get("temperature"),
                                               params.get("diameter"));
  std::cout << "Mean free path: " << gasMFP << " um" << std::endl;

  auto model = ps::SmartPointer<ps::SingleParticleALD<NumericType, D>>::New(
      params.get("stickingProbability"), params.get("numCycles"),
      params.get("growthPerCycle"), params.get("totalCycles"),
      params.get("coverageTimeStep"), params.get("evFlux"),
      params.get("inFlux"), params.get("s0"), gasMFP);

  ps::AtomicLayerProcess<NumericType, D> ALP(domain, model);
  ALP.setCoverageTimeStep(params.get("coverageTimeStep"));
  ALP.setPulseTime(params.get("pulseTime"));
  ALP.setNumCycles(params.get<unsigned>("numCycles"));
  ALP.setNumberOfRaysPerPoint(params.get<unsigned>("numRaysPerPoint"));
  ALP.disableRandomSeeds();
  ALP.apply();

  MeasureProfile<NumericType, D>(domain, params.get("gapHeight") / 2.)
      .save(params.get<std::string>("outputFile"));

  domain->saveVolumeMesh("SingleParticleALD_final");

  return 0;
}