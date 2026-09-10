// Blanket etch: the PMC's normalisation, with the geometry taken away.
//
// A flat surface has no shadowing, no reflection, and no normal to estimate,
// so the etched depth must equal the analytic blanket rate exactly -- in both
// dimensions and at every grid spacing. Anything else is a normalisation
// error, and this is the only test that isolates one.
//
// It matters because the two dimensional constants,
//
//     sites per cell    nu           = sigma0 * dx^(D-1)
//     atoms per cell    atomsPerCell = rho    * dx^D
//
// are BOTH equal to their nm-unit values at dx = 1, in 2D and in 3D alike.
// Every comparison run so far has been at dx = 1, so a wrong exponent in
// either one has been invisible. Sweeping dx separates them: a wrong power of
// dx shows up as a depth that drifts with the grid, and differs between 2D
// and 3D.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakePlane.hpp>

#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

static T EXT = 20.0;
static const T DEEP = -20.0, GAS_ABOVE = 8.0;

/// What the blanket run delivered, so the error can be split into "was the
/// flux short" and "was the removal short".
struct Audit { T depth = 0, ionsPerNm2 = 0, fAdsPerNm2 = 0, area = 0, time = 0;
               T thF = 0, thO = 0; double escFrac = 0; T areaRatio = 0; double resetPerRemoval = 0;
               T ie = 0, sp = 0, th = 0;
               T owed = 0, taken = 0, left = 0; double failFrac = 0; };

template <int D>
static Audit blanket(T dx, T target, T ER, unsigned seed, int steps,
                     const ps::ChemicalMechanism<T> &mech, bool freeze = false) {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(dx, EXT, EXT);
  ps::MakePlane<T, D>(dom, T(0), ps::Material::Si).apply();

  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D]{}, n[D]{};
    o[D - 1] = DEEP; n[D - 1] = T(1);
    ls::MakeGeometry<T, D>(deep,
        ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply(); }

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
  cells->fromLevelSets(lss, mm, GAS_ABOVE);

  cs::LatticeMap<T, D> lat(*cells);
  const auto &mid = *cells->getScalarData("Material");
  std::vector<T> fill(cells->getNumberOfCells(), T(0));
  std::vector<int> material(cells->getNumberOfCells());
  for (size_t c = 0; c < fill.size(); ++c) {
    material[c] = (int)mid[c];
    fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
  }

  // Only the INTERIOR is dosed correctly. Rays start GAS_ABOVE over the
  // surface with a cosine spread, so a border band of about that width loses
  // the rays that wander out of the domain -- through four sides in 3D and
  // two in 2D, which is the whole of the apparent dimension dependence. A
  // feature sitting in the middle of a wide domain never sees this; a
  // whole-domain average does. Measure the middle half only.
  const auto &dims = lat.dims();
  auto interior = [&](const std::array<int, D> &idx) {
    for (int d = 0; d < D - 1; ++d) {
      const T frac = (static_cast<T>(idx[d]) + T(0.5)) /
                     static_cast<T>(dims[d]);
      if (frac < T(0.25) || frac > T(0.75)) return false;
    }
    return true;
  };
  auto solidInInterior = [&](const std::vector<T> &f) {
    size_t n = 0;
    std::array<int, D> idx{};
    size_t total = 1;
    for (int d = 0; d < D; ++d) total *= static_cast<size_t>(dims[d]);
    for (size_t flat = 0; flat < total; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      if (!interior(idx)) continue;
      const int id = lat.cellId(idx);
      if (id >= 0 && f[id] >= T(0.5)) ++n;
    }
    return n;
  };
  const size_t solid0 = solidInInterior(fill);

  // Optionally kill the ion channel: with no ions there is no cascade
  // clearing, so the coverage settles on adsorption against the
  // thermal term alone. If it matches the analytic solution of that
  // same reduced system, the adsorption is right and the bias is in
  // the clearing; if it still falls short, it is in the adsorption.
  typename ps::VoxelPMC<T, D>::Parameters par;
  if (const char *e = std::getenv("BL_FLUXION"))
    par.fluxIon *= std::atof(e);
  if (std::getenv("BL_NOO2")) par.fluxO = T(0);
  if (const char *e = std::getenv("BL_S0")) par.stickF = std::atof(e);

  ps::VoxelPMC<T, D> pmc(lat, fill, material, par);
  pmc.setSeed(seed);
  pmc.setFreezeSurface(freeze);
  if (std::getenv("BL_FBAL")) pmc.setFluorineBalance(true);
  if (std::getenv("BL_HANDDOWN")) pmc.setHandDown(true);
  if (std::getenv("BL_SITES")) pmc.setSiteCounts(true);
  if (std::getenv("BL_FLUXETCH")) pmc.setFluxEtch(true);
  if (std::getenv("BL_SIMPLE")) pmc.setSimpleFlux(true);
  if (const char *e = std::getenv("BL_P")) pmc.setSimpleP(std::atof(e));
  if (std::getenv("BL_NOREEMIT")) pmc.setReemission(false);
  if (const char *e = std::getenv("BL_ARM")) pmc.setArmAfter(std::atof(e));
  if (std::getenv("BL_PLANE")) pmc.setPlaneAcceptance(true);
  if (const char *e = std::getenv("BL_PWIN")) pmc.setPlaneWindow(std::atof(e));
  // Feature toggles, to bisect which one carries the 2D/3D asymmetry.
  if (std::getenv("BL_NOREEMIT")) pmc.setReemission(false);
  if (std::getenv("BL_NOLOCAL")) pmc.setLocalClearing(false);
  if (std::getenv("BL_NOFRAC")) pmc.setFractionalRemoval(false);
  if (const char *e = std::getenv("BL_W")) pmc.setIonWeight(std::atof(e));
  if (std::getenv("BL_NOMIRROR")) pmc.setSideReflection(false);
  const T time = target / ER;
  // Coverage is time-averaged over the second half: a single end-of-run
  // snapshot on a 20 nm patch is far too noisy to compare against 0.403.
  // Coverage over the same interior window, for the same reason.
  size_t lastSurfCells = 0;   ///< exposed cells the last count saw
  auto interiorCoverage = [&]() {
    const auto &st = pmc.states();
    size_t nF = 0, nO = 0, n = 0;
    std::array<int, D> idx{};
    size_t total = 1;
    for (int d = 0; d < D; ++d) total *= static_cast<size_t>(dims[d]);
    for (size_t flat = 0; flat < total; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      if (!interior(idx)) continue;
      const int id = lat.cellId(idx);
      if (id < 0 || fill[id] < T(0.5)) continue;
      bool exposed = false;
      for (int d = 0; d < D && !exposed; ++d)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          auto nb = idx; nb[d] += sgn;
          // A neighbour OUTSIDE the lattice is not gas: counting it as gas
          // makes the buried floor of the slab look like surface, and that
          // whole layer is permanently Bare, which halves the coverage.
          const int nid = lat.cellId(nb);
          if (nid >= 0 && fill[nid] < T(0.5)) { exposed = true; break; }
        }
      if (!exposed) continue;
      ++n;
      if (st[id] == 1) ++nF; else if (st[id] == 2) ++nO;
    }
    lastSurfCells = n;
    return std::array<T, 2>{n ? T(nF) / T(n) : T(0), n ? T(nO) / T(n) : T(0)};
  };
  T sF = 0, sO = 0, sArea = 0; int nAvg = 0;
  for (int s = 0; s < steps; ++s) {
    pmc.step(time / steps);
    if (s >= steps / 2) {
      const auto c = interiorCoverage();
      sF += c[0]; sO += c[1]; ++nAvg;
      // exposed surface cells per flat-surface cell: 1.0 is flat
      const T flatCells = std::pow(EXT / 2 / dx, D - 1);
      sArea += static_cast<T>(lastSurfCells) / flatCells;
    }
  }

  // Mean recession = removed volume / lateral area. On a blanket this IS the
  // etch depth, with no surface reconstruction in the way.
  const T lateral = std::pow(EXT / 2, D - 1);   // the interior window
  Audit a;
  a.area = lateral; a.time = time;
  const size_t solid1 = solidInInterior(fill);
  a.depth = static_cast<T>(solid0 - solid1) * std::pow(dx, D) / lateral;
  a.ionsPerNm2 = static_cast<T>(pmc.nIons) / (lateral * time);
  a.fAdsPerNm2 = static_cast<T>(pmc.nAdsF) / (lateral * time);
  // measured removal probability per hit, against the intended p
  a.escFrac = pmc.nHitN ? double(pmc.removedCells()) / double(pmc.nHitN) : 0.0;
  std::cout << "      channels: total " << pmc.removedCells()
            << "  ion-enh " << pmc.remIE << "  sputter " << pmc.remSp
            << "  thermal " << pmc.remTh << "   hits " << pmc.nHitN
            << "  launches " << pmc.nLaunch << "\n";
  a.resetPerRemoval = pmc.removedCells()
      ? double(pmc.nBareReset) / double(pmc.removedCells()) : 0.0;
  // Removal by channel, as depth per second, so 2D and 3D compare
  // directly against the analytic rate regardless of cell size.
  const T perDepth = std::pow(dx, D) / (lateral * time);
  a.ie = static_cast<T>(pmc.remIE) * perDepth;
  a.sp = static_cast<T>(pmc.remSp) * perDepth;
  a.th = static_cast<T>(pmc.remTh) * perDepth;
  // The ledger, all three as an equivalent depth in nm.
  const typename ps::VoxelPMC<T, D>::Parameters pp{};
  a.owed = static_cast<T>(pmc.atomsOwed) / (pp.rho * lateral);
  a.taken = static_cast<T>(pmc.removedCells()) * std::pow(dx, D) / lateral;
  a.left = static_cast<T>(pmc.outstandingCredit()) / (pp.rho * lateral);
  a.failFrac = pmc.removedCells()
      ? double(pmc.nRemoveFail) / double(pmc.removedCells()) : 0.0;
  a.areaRatio = nAvg ? sArea / nAvg : T(0);
  a.thF = nAvg ? sF / nAvg : T(0);
  a.thO = pmc.nLaunch ? double(pmc.nHitN) / double(pmc.nLaunch) : 0.0;
  std::cout << "      removals on first hit " << pmc.nRem0
            << ", after a reflection " << pmc.nRemB
            << "   -> " << (pmc.nRem0 + pmc.nRemB ?
                 100.0 * pmc.nRemB / (pmc.nRem0 + pmc.nRemB) : 0.0)
            << " % of removals come from reflected particles\n";
  return a;
}

int main(int argc, char **argv) {
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");
  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");
  // BL_S0: the F sticking coefficient, in the mechanism as well as the PMC,
  // so the analytic target moves with it.
  if (const char *e = std::getenv("BL_S0")) {
    const T s0 = std::atof(e);
    // the sticking lives on the ADSORPTION REACTION's prefactor
    for (auto &r : mech.reactions)
      if (r.isAdsorption && r.equation.find("F +") != std::string::npos)
        r.prefactor = s0;
    for (size_t i = 0; i < mech.gas.size(); ++i)
      if (mech.gas[i].label == "F_flux")
        mech.setSticking(static_cast<int>(i), s0, mech.gas[i].stickingEa,
                         mech.gas[i].stickingFreeSiteExponent,
                         mech.gas[i].stickingBeta);
  }
  const auto gam = mech.sourceFluxes(ps::Material::Si);
  const auto kc = mech.rateConstantsFor(ps::Material::Si);
  std::vector<T> th(mech.coverageNames.size(), T(0));
  auto gEff = gam;              // the fluxes the run actually uses
  {
    auto &g = gEff;
    if (const char *e = std::getenv("BL_FLUXION")) {
      const T f = std::atof(e);
      for (size_t i = 0; i < mech.gas.size() && i < g.size(); ++i)
        if (mech.gas[i].isIonChannel) g[i] *= f;
    }
    if (std::getenv("BL_NOO2"))
      for (size_t i = 0; i < mech.gas.size() && i < g.size(); ++i)
        if (mech.gas[i].label == "O_flux") g[i] = T(0);
    mech.solveCoverages(g, kc, th);
  }
  // the rate must be priced with the SAME fluxes the coverages saw
  const T ER = std::abs(mech.growthRate(gEff, kc, th, ps::Material::Si));

  const T target = argc > 1 ? std::atof(argv[1]) : 10.0;
  const int steps = argc > 2 ? std::atoi(argv[2]) : 1500;
  const unsigned seed = argc > 3 ? (unsigned)std::atoi(argv[3]) : 7;

  std::cout << std::fixed << std::setprecision(3)
            << "blanket etch,  analytic ER " << ER << " nm/s,  target "
            << target << " nm,  " << steps << " steps, seed " << seed
            << ",  extent " << EXT << " nm\n\n"
            << "   D    dx      etched      target      error\n";
  // Expected areal arrival rates, from the mechanism itself. The reaction
  // file quotes a flux per SITE, so the areal rate is that times sigma0.
  const auto pp = ps::VoxelPMC<T, 3>::Parameters{};
  const T sigma0 = pp.sigma0;
  // nIons counts EMITTED ion events, and the ion weight multiplies those
  // while dividing the yield each carries, so the counter must be compared
  // against weight * flux * sigma0 -- not against the physical rate.
  const T ionWeight = 4;
  const T ionExpect = pp.fluxIon * sigma0 * ionWeight;
  std::cout << "   ion flux " << pp.fluxIon * sigma0 << "/nm^2/s physical, x"
            << ionWeight << " weight = " << ionExpect << " emitted;  F flux "
            << pp.fluxF * sigma0 << "/nm^2/s\n";
  std::cout << "   analytic steady-state coverages:";
  for (size_t i = 0; i < th.size(); ++i)
    std::cout << "  " << mech.coverageNames[i] << " " << th[i];
  std::cout << "\n\n"
            << "   D   ext    dx    etched     err   hits/particle"
            << "  area/flat  removals/hit\n";
  auto row = [&](int Dn, T dx, const Audit &a) {
    std::cout << std::setw(4) << Dn << std::setw(6) << std::setprecision(0) << EXT
              << std::setw(6) << std::setprecision(2) << dx
              << std::setw(10) << std::setprecision(3) << a.depth
              << std::setw(9) << std::setprecision(1)
              << 100 * (a.depth / target - 1) << " %"
              << std::setw(13) << std::setprecision(4) << a.thO
              << std::setw(11) << std::setprecision(3) << a.areaRatio
              << std::setw(14) << std::setprecision(6) << a.escFrac
              << std::setprecision(3) << "\n";
  };
  EXT = 80;
  // BL_2DONLY: a flat-surface rate check needs no 3D box and no frozen run
  const bool only2d = std::getenv("BL_2DONLY") != nullptr;
  // BL_DX: one spacing, 2D. nu = sigma0*dx^(D-1), so dx = 0.1 nm gives nu = 1
  // in 2D -- a cell IS a site, and every mapping factor becomes unity.
  if (const char *e = std::getenv("BL_DX")) {
    const T dx = std::atof(e);
    std::cout << "   nu = sigma0*dx^(D-1) = " << 10.0 * dx << "\n";
    for (unsigned sd : {seed, seed + 11u, seed + 23u})
      row(2, dx, blanket<2>(dx, target, ER, sd, steps, mech));
    std::cout << "\n   frozen surface (no removal):\n";
    row(2, dx, blanket<2>(dx, target, ER, seed, steps, mech, true));
    return 0;
  }
  // BL_DXSWEEP: 2D at three grid spacings. nu = sigma0*dx and the atoms per
  // cell are rho*dx^2, so both the coverage quantum and the material each
  // removal exposes scale with dx.
  if (std::getenv("BL_DXSWEEP")) {
    for (T dx : {T(2.0), T(1.0), T(0.5)})
      for (unsigned sd : {seed, seed + 11u})
        row(2, dx, blanket<2>(dx, target, ER, sd, steps, mech));
    std::cout << "\n   frozen surface (no removal):\n";
    for (T dx : {T(2.0), T(1.0), T(0.5)})
      row(2, dx, blanket<2>(dx, target, ER, seed, steps, mech, true));
    return 0;
  }
  for (unsigned sd : {seed, seed + 11u, seed + 23u}) {
    row(2, T(1.0), blanket<2>(T(1.0), target, ER, sd, steps, mech));
    if (!only2d)
      row(3, T(1.0), blanket<3>(T(1.0), target, ER, sd, steps, mech));
  }

  std::cout << "\n   frozen surface (no removal):\n";
  for (unsigned sd : {seed, seed + 11u}) {
    row(2, T(1.0), blanket<2>(T(1.0), target, ER, sd, steps, mech, true));
    if (!only2d)
      row(3, T(1.0), blanket<3>(T(1.0), target, ER, sd, steps, mech, true));
  }
  return 0;
}
