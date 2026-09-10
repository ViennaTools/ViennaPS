// Neutral supply to the trench floor, voxel arm against the PMC, on the
// IDENTICAL INITIAL geometry -- no etching, so no roughness has developed and
// the two surfaces are the same staircase. Any difference in coverage is
// therefore transport, not representation.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psVoxelChemistry.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeTrench.hpp>
#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <iostream>
namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 2;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif
static T DX = 1.0;
static const T W = 40.0, MASKH = 30.0;

int main(int argc, char **argv) {
  const T DEPTH = argc > 1 ? std::atof(argv[1]) : 0.0;
  if (argc > 2) DX = std::atof(argv[2]);
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm"); ps::units::Time::setUnit("s");
  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, T(4) * W, T(4) * W);
  ps::MakeTrench<T, D>(dom, W, DEPTH, T(0), MASKH, T(0), false,
                       ps::Material::Si, ps::Material::Mask).apply();
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., -36. - DEPTH}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deep, ls::SmartPointer<ls::Plane<T,D>>::New(o,n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
  auto mm = ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::Si);
  for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto cells = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
  cells->setCellSetPosition(true);
  cells->setCoverMaterial((int)ps::Material::GAS);
  cells->fromLevelSets(lss, mm, MASKH + DEPTH + T(4));
  cs::LatticeMap<T, D> lat(*cells);
  const auto &mid = *cells->getScalarData("Material");
  std::vector<T> fill(cells->getNumberOfCells(), T(0));
  std::vector<int> material(cells->getNumberOfCells());
  for (size_t c = 0; c < fill.size(); ++c) {
    material[c] = (int)mid[c];
    fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
  }
  // ---- voxel arm: converge the coverages on this geometry
  auto fillV = fill; auto matV = material;
  ps::VoxelChemistry<T, D> vox(mech, lat, fillV, matV);
  vox.setRaysPerCell(2000);
  vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
  auto cov = vox.makeCoverages();
  vox.initialiseCoverages(cov, 1000u, 200, T(1e-8));
  // ---- PMC: hold the geometry still and let the states equilibrate
  auto fillP = fill; auto matP = material;
  ps::VoxelPMC<T, D> pmc(lat, fillP, matP);
  pmc.setSeed(7);
  pmc.setNormalEstimator(cs::NormalEstimator::InterfaceAverage);
  pmc.setFreezeSurface(true);              // adsorb and desorb, never remove
  // time-average: one snapshot of ~36 binary cells has sigma ~ 0.08, which
  // is the whole effect being looked for
  const auto &stRef = pmc.states();
  std::vector<double> acc(fill.size(), 0.0);
  int nAcc = 0;
  for (int i = 0; i < 4000; ++i) {
    pmc.step(2e-5);
    if (i >= 2000) {
      for (size_t c = 0; c < fill.size(); ++c)
        acc[c] += (stRef[c] == 1) ? 1.0 : 0.0;
      ++nAcc;
    }
  }
  for (auto &a : acc) a /= std::max(nAcc, 1);
  const auto &st = pmc.states();
  const auto &dims = lat.dims();
  std::cout << std::fixed << std::setprecision(3)
            << "  theta_F on the floor of a " << DEPTH << " nm deep trench\n"
            << "  (no etching: any difference is TRANSPORT)\n\n"
            << "     x [nm]    voxel arm    PMC\n";
  for (int i = 0; i < dims[0]; ++i) {
    const T x = lat.minCorner()[0] + DX * (i + T(0.5));
    if (std::abs(x) > 21) continue;
    for (int k = dims[1] - 1; k >= 0; --k) {
      const int id = lat.cellId({i, k});
      if (id < 0 || fill[id] < T(0.5) ||
          material[id] == (int)ps::Material::Mask) continue;
      if (((int)(x + 100)) % 3 == 0)
        std::cout << "   " << std::setw(7) << x << std::setw(11) << cov[id][0]
                  << std::setw(9) << (st[id] == 1 ? 1.0 : 0.0) << "\n";
      break;
    }
  }
  // band averages
  T sv = 0, sp = 0; int n = 0;
  for (int i = 0; i < dims[0]; ++i) {
    const T x = lat.minCorner()[0] + DX * (i + T(0.5));
    if (std::abs(x) > 18) continue;
    for (int k = dims[1] - 1; k >= 0; --k) {
      const int id = lat.cellId({i, k});
      if (id < 0 || fill[id] < T(0.5) ||
          material[id] == (int)ps::Material::Mask) continue;
      sv += cov[id][0]; sp += acc[id]; ++n; break;
    }
  }
  std::cout << "    acceptance: " << pmc.nAcc << " tests, mean raw value "
            << (pmc.nAcc ? pmc.sumAccRaw / pmc.nAcc : 0.0) << ", clamped at 1 in "
            << (pmc.nAcc ? 100.0 * pmc.nAccClamped / pmc.nAcc : 0.0) << " % of them\n";
  std::cout << "\n  floor mean |x|<18:  voxel " << sv / n << "   PMC " << sp / n
            << "   analytic blanket 0.403\n";
}
