// The binary-cell particle Monte Carlo against its own analytic rate.
//
// A BLANKET wafer, because a flat surface has an absolute answer: the
// continuum steady state of the same reaction network. Nothing here is fitted
// to it -- the site states are flipped by arriving particles and by thermal
// firings, and theta is only ever COUNTED. If the etch rate lands on the
// analytic value, the mapping is right.
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>

#include <csDenseCellSet.hpp>
#include <csGridTraversal.hpp>
#include <lsMakeGeometry.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

namespace ps = viennaps;
namespace cs = viennacs;
namespace ls = viennals;
using T = double;
constexpr int D = 2;

int main() {
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);

  using PMC = ps::VoxelPMC<T, D>;
  PMC::Parameters p;

  // ---- the continuum steady state of the same network, as the target
  const T Ysp = p.A_sp * (std::sqrt(p.meanEnergy) - std::sqrt(p.Eth_sp));
  const T Yie = p.A_ie * (std::sqrt(p.meanEnergy) - std::sqrt(p.Eth_ie));
  const T Yp = p.A_p * (std::sqrt(p.meanEnergy) - std::sqrt(p.Eth_p));
  const T a = p.stickF * p.fluxF / (4 * p.kSigma + 2 * Yie * p.fluxIon);
  const T b = p.stickO * p.fluxO / (p.betaSigma + Yp * p.fluxIon);
  const T freeF = 1 / (1 + a + b);
  const T thF = a * freeF, thO = b * freeF;
  // the bracket is a per-SITE rate; multiply by sigma0 for an areal one
  const T ER =
      (p.kSigma * thF + Yie * p.fluxIon * thF + Ysp * p.fluxIon) * p.sigma0 / p.rho;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "continuum steady state (nothing is fitted to it)\n"
            << "    Y_sp " << Ysp << "  Y_ie " << Yie << "  Y_p " << Yp << "\n"
            << "    theta_F " << thF << "   theta_O " << thO
            << "   ER " << ER << " nm/s\n\n";

  for (T dx : {2.0, 1.0}) {
    // ---- a flat silicon substrate, as cells
    auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(dx, T(80), T(80));
    ps::MakePlane<T, D>(dom, dx, T(80), T(80), T(0), false, ps::Material::Si)
        .apply();
    auto top = dom->getLevelSets().back();
    auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
    {
      T o[D] = {0., -30.}, n[D] = {0., 1.};
      ls::MakeGeometry<T, D>(
          deep, ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply();
    }
    std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
    auto mm = ls::SmartPointer<ls::MaterialMap>::New();
    mm->insertNextMaterial((int)ps::Material::Si);
    for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
      lss.push_back(dom->getLevelSets()[l]);
      mm->insertNextMaterial(
          (int)dom->getMaterialMap()->getMaterialAtIdx(l));
    }
    auto cellSet = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
    cellSet->setCellSetPosition(true);
    cellSet->setCoverMaterial((int)ps::Material::GAS);
    cellSet->fromLevelSets(lss, mm, T(10));

    cs::LatticeMap<T, D> lat(*cellSet);
    const auto &mid = *cellSet->getScalarData("Material");
    std::vector<T> fill(cellSet->getNumberOfCells(), T(0));
    std::vector<int> material(cellSet->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }

    PMC pmc(lat, fill, material, p);
    pmc.setSeed(7);

    const T target = 10.0;          // nm of etch
    const T time = target / std::abs(ER);
    const int steps = 3000;
    const T dt = time / steps;
    for (int s = 0; s < steps; ++s)
      pmc.step(dt);

    const auto &dims = lat.dims();
    const T columns = static_cast<T>(dims[0]);
    const T depth = pmc.removedCells() * dx / columns;
    const auto th = pmc.coverages();
    const T secs = time;
    std::cout << std::setprecision(0)
              << "    events: adsF " << pmc.nAdsF << "  clearF " << pmc.nClearF
              << "  thermF " << pmc.nThermF << "  ions " << pmc.nIons
              << " (on F " << pmc.nIonsOnF << ")  removed "
              << pmc.removedCells() << "\n"
              << "    per second: adsF " << pmc.nAdsF / secs
              << "   clearF+thermF " << (pmc.nClearF + pmc.nThermF) / secs
              << "\n";

    std::cout << "dx = " << std::setprecision(1) << dx << " nm"
              << "   atoms/cell " << std::setprecision(1)
              << p.rho * std::pow(dx, D) << "\n"
              << std::setprecision(4)
              << "    theta_F " << th[0] << " (eq " << thF << ")"
              << "   theta_O " << th[1] << " (eq " << thO << ")\n"
              << "    depth   " << depth << " nm (target " << target << ")"
              << "   ER " << depth / time << " nm/s"
              << "   err " << std::setprecision(2)
              << 100 * (depth / target - 1) << " %\n\n";
  }

  // ---------------------------------------------------------------- trench
  // The same masked trench the other two arms run, written as a VTU so the
  // three surfaces can be laid over one another. Fill is binary here, so the
  // VTU shows a staircase where the fill-fraction arm shows a graded band.
  {
    const T dx = 1.0, W = 40.0, MASKH = 30.0;
    auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(dx, T(4) * W, T(4) * W);
    ps::MakeTrench<T, D>(dom, W, T(0), T(0), MASKH, T(0), false,
                         ps::Material::Si, ps::Material::Mask)
        .apply();
    auto top = dom->getLevelSets().back();
    auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
    {
      T o[D] = {0., -36.}, n[D] = {0., 1.};
      ls::MakeGeometry<T, D>(
          deep, ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply();
    }
    std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
    auto mm = ls::SmartPointer<ls::MaterialMap>::New();
    mm->insertNextMaterial((int)ps::Material::Si);
    for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
      lss.push_back(dom->getLevelSets()[l]);
      mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
    }
    auto cellSet = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
    cellSet->setCellSetPosition(true);
    cellSet->setCoverMaterial((int)ps::Material::GAS);
    cellSet->fromLevelSets(lss, mm, MASKH + T(4));

    cs::LatticeMap<T, D> lat(*cellSet);
    const auto &mid = *cellSet->getScalarData("Material");
    std::vector<T> fill(cellSet->getNumberOfCells(), T(0));
    std::vector<int> material(cellSet->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }

    // create the field BEFORE any reference into the cell data is
    // taken: adding one reallocates and would dangle them
    cellSet->addScalarData("State", 0.);
    auto dump = [&](const std::vector<std::uint8_t> &st,
                    const std::string &name) {
      auto &ff = *cellSet->getFillingFractions();
      auto &mmv = *cellSet->getScalarData("Material");
      auto &state = *cellSet->getScalarData("State");
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        // Material stays the material; the chemical state is its own field,
        // and the writer leaves the gas out by itself.
        // State is the field to colour by: material AND chemistry in one.
        //   0 bare Si | 1 fluorinated | 2 oxidised | 3 mask
        // Material stays the plain ViennaPS id, for analysis.
        const bool solid = fill[c] >= T(0.5);
        const bool mask = material[c] == (int)ps::Material::Mask;
        mmv[c] = solid ? T(material[c]) : T((int)ps::Material::GAS);
        state[c] = !solid ? T(0) : (mask ? T(3) : T(st[c]));
      }
      cellSet->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };

    PMC pmc(lat, fill, material, p);
    pmc.setSeed(7);
    dump(pmc.states(), "pmc_trench_initial.vtu");

    const T time = 10.0 / std::abs(ER);
    const int steps = 3000;
    for (int s = 0; s < steps; ++s)
      pmc.step(time / steps);

    dump(pmc.states(), "pmc_trench_final.vtu");
    const auto th = pmc.coverages();
    const auto &dims = lat.dims();
    std::cout << std::setprecision(4)
              << "  trench W=40 mask=30, dx=1:  theta_F " << th[0]
              << "  theta_O " << th[1] << "   cells removed "
              << pmc.removedCells() << "  (" 
              << pmc.removedCells() * dx / static_cast<T>(dims[0])
              << " nm averaged over the full width)\n";
  }
}
