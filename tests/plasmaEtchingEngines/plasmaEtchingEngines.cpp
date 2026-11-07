#include <geometries/psMakeTrench.hpp>
#include <models/psSF6O2Etching.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// The ion fluxes of SF6/O2 etching from the CPU and the GPU triangle engine
// on 2D trenches in a mask, averaged over the trench floor, agree. With mask
// walls that lean back by a degree the ions that graze the walls reflect onto
// the floor with most of their energy, and the ion-enhanced flux (threshold
// 15 eV) and the sputter flux take them. With walls that lean back by 15
// degrees the ions hit the walls at about 75 degrees and reflect with about
// 16 eV; with a passivation threshold of 1 eV the ion-enhanced passivation
// flux takes the reflected ions above the substrate thresholds, below which
// both engines stop an ion.

using namespace viennaps;

namespace {

using T = double;
constexpr int D = 2;

std::vector<double> floorFluxes(FluxEngineType engine, T taper,
                                T passivationThreshold,
                                const std::vector<std::string> &labels) {
  const T gridDelta = 0.03, opening = 0.35, maskHeight = 1.2;
  const T extent =
      opening + 2. * maskHeight * std::tan(taper * M_PI / 180.) + 0.6;
  auto domain = Domain<T, D>::New();
  MakeTrench<T, D>(domain, gridDelta, extent, extent, opening, maskHeight,
                   taper, 0., false, true, Material::Si)
      .apply();

  auto params = SF6O2Etching<T, D>::defaultParameters();
  params.Ions.meanEnergy = 100.;
  params.Ions.sigmaEnergy = 10.;
  params.Ions.exponent = 1000.;
  params.Passivation.Eth_ie = passivationThreshold;
  auto model = SmartPointer<SF6O2Etching<T, D>>::New(params);

  RayTracingParameters rays;
  rays.raysPerPoint = 10000;
  rays.useRandomSeeds = false;
  rays.rngSeed = 7;

  Process<T, D> process(domain, model);
  process.setParameters(rays);
  process.setFluxEngineType(engine);
  process.setProcessDuration(1.);
  auto mesh = process.calculateFlux();
  VC_TEST_ASSERT(mesh);

  const auto &nodes = mesh->getNodes();
  const auto *materials = mesh->getCellData().getScalarData("MaterialIds");
  VC_TEST_ASSERT(materials);
  std::vector<double> means;
  for (const auto &label : labels) {
    const auto *flux = mesh->getCellData().getScalarData(label);
    VC_TEST_ASSERT(flux);
    double sum = 0.;
    int count = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (MaterialMap::isMaterial((*materials)[i], Material::Si) &&
          std::abs(nodes[i][D - 1]) < 0.5 * gridDelta) {
        sum += (*flux)[i];
        ++count;
      }
    }
    VC_TEST_ASSERT(count > 5);
    means.push_back(sum / count);
  }
  return means;
}

void compareEngines(T taper, T passivationThreshold,
                    const std::vector<std::string> &labels) {
  const auto cpu = floorFluxes(FluxEngineType::CPU_TRIANGLE, taper,
                               passivationThreshold, labels);
  const auto gpu = floorFluxes(FluxEngineType::GPU_TRIANGLE, taper,
                               passivationThreshold, labels);
  for (size_t k = 0; k < labels.size(); ++k) {
    std::cout << "Mask walls leaning back by " << taper << " degrees, "
              << labels[k] << " on the floor: CPU " << cpu[k] << ", GPU "
              << gpu[k] << ", ratio " << gpu[k] / cpu[k] << std::endl;
    VC_TEST_ASSERT(std::abs(gpu[k] / cpu[k] - 1.) < 0.01);
  }
}

} // namespace

int main() {
  if (!gpuAvailable()) {
    std::cout << "No GPU available, test skipped." << std::endl;
    return 0;
  }
  Logger::setLogLevel(LogLevel::WARNING);
  units::Length::setUnit("um");
  units::Time::setUnit("min");

  // grazing reflections at nearly vertical mask walls
  compareEngines(1.2, 10., {"ionEnhancedFlux", "ionSputterFlux"});
  // reflections at about 75 degrees, which leave the ions near the substrate
  // thresholds
  compareEngines(15., 1., {"ionEnhancedPassivationFlux"});
  return 0;
}
