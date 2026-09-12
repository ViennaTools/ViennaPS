// The C2F6/SiO2 fluorocarbon mechanism of Zhang & Kushner, JVST A 19, 524
// (2001), in the binary-cell PMC.
//
// The paper's equipment-scale model regulates every wafer reaction with a
// transfer coefficient, Eq. (5), lambda = 1/(1 + 0.6[P] + 0.1[P]^2), where [P]
// is the polymer thickness in monolayers. Their feature-scale model does not
// and says why: the polymer is a STACK OF CELLS, and a thicker stack shields
// the wafer by being in the way. That is this arm, so no lambda appears
// anywhere in the model -- the effective transfer coefficient is measured from
// the run instead.
//
//   ./voxelFC [bias_V] [seconds] [dx_nm] [seed]
//
//   FC_SWEEP=1     sweep the self-bias from 20 to 160 V
//   FC_W           wafer width in nm
//   FC_RANGE       mean ion penetration per eV, nm (default 0.0128)
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakePlane.hpp>
#include <psUnits.hpp>

#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <vector>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 2;

static T DX = 0.3;      // one adsorption site, and half a polymer monolayer
static T W = 60.0;      // wafer width, nm
static T HEIGHT = 40.0; // gas headroom + substrate, nm

static ps::SmartPointer<cs::DenseCellSet<T, D>> makeCells() {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, W, T(2) * HEIGHT);
  ps::MakePlane<T, D>(dom, T(0), ps::Material::SiO2).apply();
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., -HEIGHT + T(4)}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deep,
        ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep, top};
  auto mm = ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::SiO2);
  mm->insertNextMaterial((int)ps::Material::SiO2);
  auto c = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
  c->setCellSetPosition(true);
  c->setCoverMaterial((int)ps::Material::GAS);
  c->fromLevelSets(lss, mm, HEIGHT);
  return c;
}

struct Result { T monolayers, rate, lambdaEff; size_t through, stopped; };

static Result run(T bias, T seconds, unsigned seed) {
  auto cells = makeCells();
  cs::LatticeMap<T, D> lat(*cells);
  const auto &mid = *cells->getScalarData("Material");
  std::vector<T> fill(cells->getNumberOfCells(), T(0));
  std::vector<int> material(cells->getNumberOfCells());
  for (size_t c = 0; c < fill.size(); ++c) {
    material[c] = (int)mid[c];
    fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
  }
  cells->addScalarData("State", 0.);

  ps::VoxelPMC<T, D>::Parameters p;
  p.cosinePowerIon = 500;
  ps::VoxelPMC<T, D> pmc(lat, fill, material, p);
  typename ps::VoxelPMC<T, D>::FCParameters f;
  f.meanEnergy = bias + 14.0;      // sheath drop: Vs = 84 V gives Vd = 98 V
  if (const char *e = std::getenv("FC_RANGE")) f.rangePerEv = std::atof(e);
  if (const char *e = std::getenv("FC_NRANGE")) f.neutralRange = std::atof(e);
  if (std::getenv("FC_REDUCED")) f.fullTable = false;
  // GROWTH ONLY: every consumption channel off, so what is left is the film
  // the deposition path alone builds.
  if (std::getenv("FC_GROWONLY")) {
    f.pFPoly = 0; f.pSputterP = 0; f.pPolyWafer = 0; f.pChemSputter = 0;
    f.pDissociate = 0; f.pFluor1 = 0; f.pFluor2 = 0; f.pPassivate = 0;
    f.fluxIon = 0; f.fluxF = 0;
    f.fluxCF3p = f.fluxCF2p = f.fluxC2F4p = f.fluxC2F5p = f.fluxArp = 0;
  }
  pmc.setSeed(seed);
  if (std::getenv("FC_NOREEMIT")) pmc.setReemission(false);
  if (const char *e = std::getenv("FC_BOUNCE")) pmc.setMaxBounce(std::atoi(e));
  if (const char *e = std::getenv("FC_ARM")) pmc.setArmAfter(std::atof(e));
  pmc.setFluorocarbon(true, f);

  const int steps = std::getenv("FC_STEPS") ? std::atoi(std::getenv("FC_STEPS"))
                                            : 400;
  double oxAtHalf = 0;
  size_t ionThroughHalf = 0, ionStoppedHalf = 0;
  const bool trace = std::getenv("FC_TRACE") != nullptr;
  for (int s = 0; s < steps; ++s) {
    pmc.step(seconds / steps);
    if (trace && (s + 1) % (steps / 8) == 0)
      std::cout << "        t = " << std::setw(6)
                << seconds * (s + 1) / steps << " s   [P] = "
                << pmc.polymerMonolayers() << "\n" << std::flush;
    if (s == steps / 2 - 1) {
      oxAtHalf = pmc.oxAtomsRemoved;
      ionThroughHalf = pmc.nIonThrough;
      ionStoppedHalf = pmc.nIonStopped;
    }
  }
  if (std::getenv("FC_VTU")) {
    auto &ff = *cells->getFillingFractions();
    auto &mmv = *cells->getScalarData("Material");
    auto &st = *cells->getScalarData("State");
    const auto &pst = pmc.states();
    for (size_t c = 0; c < fill.size(); ++c) {
      ff[c] = fill[c];
      const bool solid = fill[c] >= T(0.5);
      mmv[c] = solid ? T(material[c]) : T((int)ps::Material::GAS);
      st[c] = solid ? T(pst[c]) : T(0);
    }
    char nm[64];
    std::snprintf(nm, sizeof nm, "fc_%03d.vtu", (int)std::lround(bias));
    cells->writeVTU(nm);
    std::cout << "      wrote " << nm << "\n";
  }
  // measured over the second half only, once the film has reached steady state
  // from the ATOMS taken, not the cells flipped: a cell is 2 SiO2 units at
  // dx = 0.3 nm, so a cell count is the same number rounded, and most of the
  // noise
  const T removed = T(pmc.oxAtomsRemoved - oxAtHalf);
  const T rate = removed / f.rhoOxide / (W * seconds / 2) * 60.0;   // nm/min
  const size_t thr = pmc.nIonThrough - ionThroughHalf;
  const size_t stp = pmc.nIonStopped - ionStoppedHalf;
  // the transfer coefficient the paper needs as an input, measured as an
  // output: the fraction of ions that reach the wafer through the film
  const T lam = (thr + stp) ? T(thr) / T(thr + stp) : T(1);
  std::cout << "      launched " << pmc.nLaunch << ", grown " << pmc.nPolyGrown
            << ", F-etched " << pmc.nPolyFEtch << ", sputtered "
            << pmc.nPolySputter << ", P-wafer " << pmc.nPolyWafer
            << ", activated " << pmc.nPolyAct
            << ", passivated " << pmc.nOxPassivate
            << ", Si events " << pmc.nOxEvents
            << " (ion " << pmc.nSiIon << " / F " << pmc.nSiF
            << " / P-wafer " << pmc.nSiPolyWafer << ")"
            << "\n      carbon ledger: in " << pmc.nPolyGrown << ", out "
            << std::llround(pmc.polyAtomsRemoved)
            << " (events: F " << pmc.nPolyFEtch << " + sputter "
            << pmc.nPolySputter << " + P-wafer " << pmc.nPolyWafer
            << "), net "
            << ((long long)pmc.nPolyGrown -
                (long long)std::llround(pmc.polyAtomsRemoved))
            << ", HELD " << (long long)std::llround(pmc.polymerCarbons())
            << "  (leak "
            << ((long long)pmc.nPolyGrown -
                (long long)std::llround(pmc.polyAtomsRemoved) -
                (long long)std::llround(pmc.polymerCarbons()))
            << ")"
            << ", reflections/particle "
            << double(pmc.nBounce) / std::max<size_t>(pmc.nLaunch, 1)
            << ", stick/launch " << double(pmc.nPolyGrown) / std::max<size_t>(pmc.nLaunch, 1)
            << "\n";
  return {pmc.polymerMonolayers(), rate, lam, thr, stp};
}

int main(int argc, char **argv) {
  T bias = 84.0, seconds = 4.0;
  unsigned seed = 7;
  if (argc > 1) bias = std::atof(argv[1]);
  if (argc > 2) seconds = std::atof(argv[2]);
  if (argc > 3) DX = std::atof(argv[3]);
  if (argc > 4) seed = (unsigned)std::atoi(argv[4]);
  if (const char *e = std::getenv("FC_W")) W = std::atof(e);
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");

  std::cout << std::fixed << std::setprecision(3)
            << "C2F6 / SiO2 blanket, dx = " << DX << " nm, W = " << W
            << " nm, t = " << seconds << " s\n"
            << "  -Vs[V]   E[eV]     [P] ML    rate nm/min   lambda_eff  "
               "ions through / stopped\n";
  std::vector<T> biases;
  if (std::getenv("FC_SWEEP"))
    biases = {20, 40, 60, 84, 100, 120, 140, 160};
  else
    biases = {bias};
  for (T b : biases) {
    const auto r = run(b, seconds, seed);
    std::cout << "  " << std::setw(6) << b << std::setw(8) << b + 14
              << std::setw(11) << r.monolayers << std::setw(15) << r.rate
              << std::setw(13) << r.lambdaEff << "   " << r.through << " / "
              << r.stopped << "\n" << std::flush;
  }
}
