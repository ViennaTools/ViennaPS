#include <geometries/psMakeTrench.hpp>
#include <models/psSingleParticleALD.hpp>
#include <process/psProcess.hpp>
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

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

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

  ps::SingleParticleALDParams alpModelParams;
  alpModelParams.stickingProbability = params.get("stickingProbability");
  alpModelParams.gasMeanFreePath = gasMFP;
  alpModelParams.growthPerCycle = params.get("growthPerCycle");
  alpModelParams.totalCycles = params.get<int>("totalCycles");
  alpModelParams.numCycles = params.get<int>("numCycles");
  alpModelParams.evaporationFlux = params.get("evFlux");
  alpModelParams.incomingFlux = params.get("inFlux");
  alpModelParams.s0 = params.get("s0");
  alpModelParams.purgePulseTime = params.get("purgePulseTime");
  alpModelParams.coverageDiffusionCoefficient = 100.0;

  auto model = ps::SmartPointer<ps::SingleParticleALD<NumericType, D>>::New(
      alpModelParams);

  ps::AtomicLayerProcessParameters alpParams;
  alpParams.numCycles = params.get<unsigned>("numCycles");
  alpParams.pulseTime = params.get("pulseTime");
  alpParams.coverageTimeStep = params.get("coverageTimeStep");
  alpParams.purgePulseTime = params.get("purgePulseTime");

  ps::RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = 1000;
  rayTracingParams.maxReflections = 100000;

  ps::Process<NumericType, D> ALP(domain, model);
  ALP.setParameters(alpParams);
  ALP.setParameters(rayTracingParams);
  ALP.apply();

  MeasureProfile<NumericType, D>(domain, params.get("gapHeight") / 2.)
      .save(params.get<std::string>("outputFile"));

  domain->saveVolumeMesh("SingleParticleALD_final");

  return 0;
}