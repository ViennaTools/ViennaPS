// The same trench, the same chemistry, the same etch time -- written out by
// all three surface representations so they can be laid over one another:
//
//   level set        an implicit surface, read as a mesh          -> .vtp
//   voxel            cells carrying a FILLING FRACTION in [0,1]   -> .vtu
//   binary-cell PMC  cells that are solid or gas, plus a STATE    -> .vtu
//
// The first two share ChemicalMechanism, so they run identical reaction rates
// and differ only in how the interface is represented and moved. The PMC
// carries the same network as discrete site states; its rate constants are the
// mechanism's, converted to per-site units by sigma0.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeTrench.hpp>
#include <process/psProcess.hpp>

#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <cmath>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 2;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

static T DX = 1.0;                 // argv[3]: grid spacing
static T W = 40.0, MASKH = 30.0;   // PMC_W / PMC_MASK override, in nm
static T TARGET = 10.0;   // nm of blanket-equivalent etch; argv[1] overrides
static unsigned SEED = 7;  // argv[2]: PMC seed, for spread over seeds

static ps::SmartPointer<ps::Domain<T, D>> makeDomain() {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, T(4) * W, T(4) * W);
  // PRETRENCH carves a PERFECT trench of the given depth, vertical walls, in
  // BOTH arms -- the cells are built from these same level sets. With the
  // surface then frozen, the two arms are compared on identical, static,
  // exactly vertical geometry, so any difference in ion flux or incidence
  // angle is transport alone: no evolved roughness, no coverage history, no
  // exposure-time confound.
  T depth = T(0);
  if (const char *e = std::getenv("PRETRENCH")) depth = std::atof(e);
  // PRETAPER tilts the wall by a known angle. The true outward normal then
  // makes that angle with the horizontal, so |n_z| = sin(taper) is an exact
  // expectation for whatever normal the PMC's estimator reports -- the
  // staircase is stressed here in a way an axis-aligned wall never is.
  T taper = T(0);
  if (const char *e = std::getenv("PRETAPER")) taper = std::atof(e);
  ps::MakeTrench<T, D>(dom, W, depth, taper, MASKH, T(0), false,
                       ps::Material::Si, ps::Material::Mask).apply();
  return dom;
}

// cells from the level sets, with a deep plane closing the bottom so the
// substrate is a finite slab rather than a half space
static ps::SmartPointer<cs::DenseCellSet<T, D>>
makeCells(ps::SmartPointer<ps::Domain<T, D>> dom) {
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., -36.}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deep,
        ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
  auto mm = ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::Si);
  for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto cs_ = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
  cs_->setCellSetPosition(true);
  cs_->setCoverMaterial((int)ps::Material::GAS);
  cs_->fromLevelSets(lss, mm, MASKH + T(4));
  return cs_;
}

static void writeSurface(ps::SmartPointer<ps::Domain<T, D>> dom,
                         const std::string &name) {
  auto mesh = ps::SmartPointer<ls::Mesh<T>>::New();
  ls::ToSurfaceMesh<T, D>(dom->getLevelSets().back(), mesh).apply();
  ls::VTKWriter<T>(mesh, name).apply();
  // and the same line segments as plain text: the .vtp is polydata, which the
  // usual python readers do not take, and this is what gets plotted
  const std::string csv = name.substr(0, name.rfind('.')) + ".csv";
  std::ofstream f(csv);
  f << "x0,y0,x1,y1\n";
  const auto &nodes = mesh->getNodes();
  for (const auto &l : mesh->template getElements<2>())
    f << nodes[l[0]][0] << ',' << nodes[l[0]][1] << ',' << nodes[l[1]][0] << ','
      << nodes[l[1]][1] << '\n';
  std::cout << "    wrote " << name << " and " << csv << "\n";
}

int main(int argc, char **argv) {
  if (argc > 1)
    TARGET = std::atof(argv[1]);
  if (argc > 2)
    SEED = (unsigned)std::atoi(argv[2]);
  if (argc > 3)
    DX = std::atof(argv[3]);
  if (const char *e = std::getenv("PMC_W")) W = std::atof(e);
  if (const char *e = std::getenv("PMC_MASK")) MASKH = std::atof(e);
  const bool pmcOnly = argc > 4;
  ps::Logger::setLogLevel(std::getenv("LS_FLUX") ? ps::LogLevel::INTERMEDIATE
                                                 : ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");
  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");

  // NO_ION: drop the ion channel from the mechanism, so every arm runs the
  // purely spontaneous etch. The rate is recomputed from it, so the etch
  // time follows automatically.
  // NO_O2 drops the oxygen too: with no ions to clear it the surface simply
  // passivates, so the spontaneous case is F + Si -> SiF4 alone.
  if (std::getenv("NO_ION"))
    for (auto &g : mech.gas)
      if (g.isIonChannel)
        g.sourceFlux = T(0);
  if (std::getenv("NO_O2"))
    for (auto &g : mech.gas)
      if (g.label == "O_flux")
        g.sourceFlux = T(0);
  // NO_F: drop fluorine entirely. NO_IE: drop the ion-ENHANCED yield channels
  // and keep physical sputtering, whose angular form is the other branch.
  if (std::getenv("NO_F"))
    for (auto &g : mech.gas)
      if (g.label == "F_flux")
        g.sourceFlux = T(0);
  if (std::getenv("NO_IE"))
    for (auto &y : mech.ionYields)
      if (y.enhanced) {
        y.A = T(0);
        y.materialA = ps::MaterialValueMap<T>::fromDefault(T(0));
      }

  // NOREFL_BOTH: switch ion reflection off in EVERY arm, so what reaches a
  // surface is only what came straight from the source. The level set reflects
  // through thetaRMin (sticking = 1 - sat((theta-thetaRMin)/(thetaRMax-thetaRMin)));
  // pushing thetaRMin to 90 deg means an ion always sticks.
  if (std::getenv("NOREFL_BOTH")) {
    mech.ionSource.thetaRMin = T(90);
    mech.ionSource.thetaRMax = T(90);
    std::cout << "  [ion reflection OFF in every arm]\n";
  }
  // blanket-equivalent etch time, from the mechanism's own steady state
  const auto gam = mech.sourceFluxes(ps::Material::Si);
  const auto kc = mech.rateConstantsFor(ps::Material::Si);
  std::vector<T> th(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gam, kc, th);
  const T ER = std::abs(mech.growthRate(gam, kc, th, ps::Material::Si));
  const T time = TARGET / ER;
  std::cout << std::fixed << std::setprecision(4)
            << "trench W=" << W << " mask=" << MASKH << " dx=" << DX
            << ",  blanket ER " << ER << " nm/s,  t = " << time << " s ("
            << TARGET << " nm blanket-equivalent)\n\n";

  // ------------------------------------------------------------- level set
  if (!pmcOnly) {
    auto dom = makeDomain();
    writeSurface(dom, "cmp_ls_initial.vtp");
    auto model = ps::SmartPointer<ps::SurfaceChemistry<T, D>>::New(mech);
    // Transient coverages are the DEFAULT for this SF6/O2 benchmark: the
    // quasi-steady-state assumption fails on the sidewall, where a removed
    // fluorinated cell exposes bare Si that can etch before oxygen re-covers
    // it. Measured at 20 nm: flare 0.098 -> 0.676 nm, closing the gap to the
    // PMC from 15x to 2.2x, at no extra cost (167.6 s vs 166.4 s) and with no
    // change on a blanket, where the QSSA is valid (-4.9511 vs -4.9478).
    // It is NOT the global default for SurfaceChemistry -- set here only.
    // LS_STEADY=1 restores the steady-state solve for comparison.
    if (!std::getenv("LS_STEADY")) {
      model->setTransientCoverages(true);
      std::cout << "  [level set: TRANSIENT coverages (LS_STEADY=1 to disable)]\n";
    }
    ps::Process<T, D> proc(dom, model, time);
    proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
    ps::RayTracingParameters rt;
    rt.raysPerPoint = 400; rt.useRandomSeeds = false; rt.rngSeed = 1000;
    proc.setParameters(rt);
    ps::CoverageParameters cov; cov.tolerance = 1e-6; cov.maxIterations = 40;
    proc.setParameters(cov);
    // LS_DTRATIO scales the advection CFL ratio, to check that a transient
    // coverage result is converged in the time step rather than an artefact
    // of taking one backward-Euler step per advection step.
    if (const char *e = std::getenv("LS_DTRATIO")) {
      ps::AdvectionParameters adv;
      adv.timeStepRatio = 0.4999 * std::atof(e);
      proc.setParameters(adv);
      std::cout << "  [level set: timeStepRatio x" << e << "]\n";
    }
    std::cout << "level set:\n";
    proc.apply();
    writeSurface(dom, "cmp_ls_final.vtp");
  }

  // --------------------------------------------------- filling-fraction voxel
  // LS_ONLY runs the level set and nothing else. A time-step or coverage
  // study varies only that arm, and the other two are the expensive ones.
  const bool lsOnly = std::getenv("LS_ONLY") != nullptr;
  if (!pmcOnly && !lsOnly) {
    auto cells = makeCells(makeDomain());
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> fill(cells->getNumberOfCells(), T(0));
    std::vector<int> material(cells->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    cells->addScalarData("State", 0.);
    cells->addScalarData("Nz", -1.);      // estimator normal, for validation   // before any reference is taken
    ps::VoxelChemistry<T, D> vox(mech, lat, fill, material);
    vox.setRaysPerCell(400);
    vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
    // VOX_SPREAD: how a cell's surplus fill is handed on -- 0 along the
    // dominant axis, 1 over the outward faces, 2 over the faces and the
    // diagonal. It is a deposition fix; this is here to check it changes
    // nothing under an etch.
    if (const char *e = std::getenv("VOX_SPREAD"))
      vox.setSurplusSpreading(std::atoi(e));
    if (const char *e = std::getenv("VOX_NGATE"))
      vox.setFacetGate(std::atof(e));
    auto cov = vox.makeCoverages();
    vox.initialiseCoverages(cov, 1000u, 100, T(1e-6));
    auto dump = [&](const std::string &name) {
      auto &ff = *cells->getFillingFractions();
      auto &mmv = *cells->getScalarData("Material");
      auto &state = *cells->getScalarData("State");
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        // the true material, so the writer can drop the gas by itself. An
        // emptied cell becomes gas; a partly filled one is still silicon,
        // and its detail lives in FillingFraction.
        mmv[c] = fill[c] <= T(1e-6) ? T((int)ps::Material::GAS)
                                    : T(material[c]);
        // same colour key as the PMC, minus the chemistry it does not have
        state[c] = material[c] == (int)ps::Material::Mask ? T(3) : T(0);
      }
      cells->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };
    std::cout << "filling-fraction voxel:\n";
    dump("cmp_voxel_initial.vtu");
    const int steps = 200;
    for (int s = 0; s < steps; ++s) vox.step(time / steps, cov, 100003u + s);
    dump("cmp_voxel_final.vtu");
  }

  // ------------------------------------------------------- binary-cell PMC
  // Twice: once rounding each event to a whole cell, once carrying the
  // remainder. Same seed, same geometry, so the difference is the rounding.
  struct Cfg { const char *tag; bool frac; T w; bool loc; bool reem;
               cs::NormalEstimator est; };
  const auto FACE = cs::NormalEstimator::Face;
  const auto YOUNGS = cs::NormalEstimator::FillGradientYoungs;
  // Mean of the outward normals of the exposed cell faces within the
  // search radius. 2.5 deg against voxelised planes, and the estimator
  // the comparison runs on.
  const auto IFACE = cs::NormalEstimator::InterfaceAverage;
  const Cfg cfgs[] = {
      {"pmcrew4", true, 4, true, true, FACE},      // face normals
      {"pmcw16", true, 16, true, true, IFACE},     // + M=16, smoother
      {"pmciface", true, 4, true, true, IFACE},    // face average
  };
  // PMC_CFG=<tag> runs just that one PMC configuration. The three exist to
  // compare normal estimators; a radius or seed sweep needs only one, and
  // skipping the others is a 3x saving on the expensive arm.
  const char *onlyCfg = std::getenv("PMC_CFG");
  for (const auto &cfg : cfgs) {
    if (lsOnly)
      break;
    if (onlyCfg && std::string(cfg.tag) != onlyCfg)
      continue;
    auto cells = makeCells(makeDomain());
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> fill(cells->getNumberOfCells(), T(0));
    std::vector<int> material(cells->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    // create the field BEFORE any reference into the cell data is
    // taken: adding one reallocates and would dangle them
    cells->addScalarData("State", 0.);
    cells->addScalarData("Nz", -1.);      // estimator normal, for validation
    ps::VoxelPMC<T, D>::Parameters p;
    if (const char *e = std::getenv("PMC_AIE")) p.A_ie *= std::atof(e);
    if (std::getenv("NO_ION")) p.fluxIon = T(0);
    if (std::getenv("NO_F")) p.fluxF = T(0);
    if (std::getenv("NO_IE")) p.A_ie = T(0);
    if (std::getenv("NO_O2")) p.fluxO = T(0);
    ps::VoxelPMC<T, D> pmc(lat, fill, material, p);
    pmc.setSeed(SEED);
    pmc.setFractionalRemoval(cfg.frac);
    pmc.setIonWeight(cfg.w);
    pmc.setLocalClearing(cfg.loc);
    pmc.setReemission(cfg.reem);
    pmc.setNormalEstimator(cfg.est);
    if (const char *e = std::getenv("PMC_FITR")) pmc.setFitRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_REFLR")) pmc.setReflectRadius(std::atoi(e));
    if (std::getenv("PMC_NOCAP")) pmc.setCurvatureCap(false);
    if (std::getenv("PMC_NOSEG")) pmc.setSegmentedFit(false);
    if (const char *e = std::getenv("PMC_AREANORM")) pmc.setAreaNormalisation(std::atoi(e));
    if (std::getenv("PMC_FBAL")) pmc.setFluorineBalance(true);
    if (const char *e = std::getenv("PMC_SEGTOL")) pmc.setSegmentTolerance(std::atof(e));
    if (std::getenv("PMC_NOREEMIT")) pmc.setReemission(false);
    // PMC_SIMPLE: the flux-only model -- a particle hits a cell and either
    // removes it or reflects. No coverage, no state, no thermal firing.
    if (std::getenv("PMC_COVERAGE")) pmc.setSimpleFlux(false);
    // PMC_OINIT: start the whole surface oxidised, to ask whether a passivated
    // sidewall is a stable state or only an artefact of the continuum arms
    // solving their coverages to steady state at every step.
    if (std::getenv("PMC_OINIT"))
      pmc.setInitialState(ps::VoxelPMC<T, D>::Oxidised);
    if (const char *e = std::getenv("PMC_P")) pmc.setSimpleP(std::atof(e));
    if (const char *e = std::getenv("PMC_BOUNCE")) pmc.setMaxBounce(std::atoi(e));
    if (std::getenv("PMC_NOIONREFL")) pmc.setIonReflection(false);
    if (std::getenv("PMC_OXPROTECT")) pmc.setProtectOxide(true);
    if (std::getenv("PMC_THERMLOCAL")) pmc.setThermalLocal(true);
    if (std::getenv("PMC_IONSPLIT")) pmc.setIonSplitRadius(true);
    if (const char *e = std::getenv("PMC_IONNORMR")) pmc.setIonNormalRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_DMGSCALE"))
      pmc.setDamageScale(std::atof(e));
    if (std::getenv("PMC_HITTALLY")) pmc.setHitTally(true);
    if (const char *e = std::getenv("PMC_THERMAREA"))
      pmc.setThermalArea(std::atoi(e));
    // PMC_FLUXF scales the F flux so a prescribed p can be raised without
    // changing the etch rate: rate = flux*dx^(D-1) * p * dx.
    if (const char *e = std::getenv("PMC_FLUXF")) p.fluxF = std::atof(e);
    if (std::getenv("PMC_NOPLANE")) pmc.setPlaneAcceptance(false);
    if (const char *e = std::getenv("PMC_PWIN")) pmc.setPlaneWindow(std::atof(e));
    if (const char *e = std::getenv("PMC_MINPTS")) pmc.setMinFitPoints(std::atoi(e));
    if (std::getenv("PMC_NOPRUNE")) pmc.setPruneIslands(false);
    if (std::getenv("PMC_OFFPLANE")) pmc.setOffPlaneTest(true);
    if (const char *e = std::getenv("PMC_IONW")) pmc.setIonWeight(std::atof(e));
    // PMC_EST=fit selects the least-squares plane through cell centres;
    // the default table entry uses the mean of the exposed faces.
    if (const char *e = std::getenv("PMC_EST"))
      pmc.setNormalEstimator(std::string(e) == "fit"
                             ? cs::NormalEstimator::InterfaceFit
                             : cs::NormalEstimator::InterfaceAverage);
    if (const char *e = std::getenv("PMC_CURVA")) pmc.setCurvatureAlpha(std::atof(e));
    if (const char *e = std::getenv("PMC_MINR")) pmc.setMinFitRadius(std::atoi(e));
    const std::string tag = cfg.tag;
    auto dump = [&](const std::string &name) {
      auto &ff = *cells->getFillingFractions();
      auto &mmv = *cells->getScalarData("Material");
      auto &state = *cells->getScalarData("State");
      auto &nzf = *cells->getScalarData("Nz");
      { std::vector<T> nz; pmc.fillNormalZ(nz);
        for (size_t c = 0; c < nz.size() && c < nzf.size(); ++c) nzf[c] = nz[c]; }
      const auto &st = pmc.states();
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        // Material stays the MATERIAL -- the chemical state is its own field,
        // so neither can alias the other (Mask is 0 and GAS is 3 in the enum)
        // State is the field to colour by: material AND chemistry in one.
        //   0 bare Si | 1 fluorinated | 2 oxidised | 3 mask
        // Material stays the plain ViennaPS id, for analysis.
        const bool solid = fill[c] >= T(0.5);
        const bool mask = material[c] == (int)ps::Material::Mask;
        mmv[c] = solid ? T(material[c]) : T((int)ps::Material::GAS);
        state[c] = !solid ? T(0) : (mask ? T(3) : T(st[c]));
      }
      cells->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };
    std::cout << "binary-cell PMC [" << tag << "]: "
              << (cfg.frac ? "fractional" : "whole-cell") << ", ion weight "
              << cfg.w << (cfg.loc ? ", local clearing" : "")
              << (cfg.reem ? ", re-emission" : "")
              << (cfg.est == YOUNGS ? ", Youngs normals"
                  : cfg.est == IFACE  ? ", interface-average normals"
                                      : ", face normals")
              << "\n";
    if (!cfg.frac)
      dump("cmp_" + tag + "_initial.vtu");
    const int steps = static_cast<int>(3000 * TARGET / 10.0);
    // Flux-only: hold the geometry for the whole run and just tally arrivals.
    const bool fluxOnly = std::getenv("PMC_FLUXONLY") != nullptr;
    if (fluxOnly) { pmc.setFreezeSurface(true); pmc.setHitTally(true); }
    // Coverage pre-equilibration, the PMC's analogue of the continuum's
    // coverageInitIterations: adsorb, desorb and ion-clear on a FROZEN
    // surface until theta stops moving, then let the surface move. The
    // continuum does this before its first advection step because the
    // adsorption timescale (1/(4 k_sigma) = 3 ms) is far shorter than the
    // etch timescale, so it starts from the converged coverage instead of
    // from a bare wafer.
    if (const char *e = std::getenv("PMC_PREEQ")) {
      const int nEq = std::atoi(e);
      pmc.setFreezeSurface(true);
      for (int s = 0; s < nEq; ++s) {
        pmc.step(time / steps);
        if (((s + 1) % std::max(1, nEq / 4)) == 0) {
          const auto ce = pmc.coverages();
          std::cout << "    pre-eq " << (s + 1) << "/" << nEq
                    << "  theta_F " << ce[0] << "  theta_O " << ce[1] << "\n";
        }
      }
      pmc.setFreezeSurface(false);
      const auto ce = pmc.coverages();
      std::cout << std::setprecision(4)
                << "    pre-equilibrated: theta_F " << ce[0]
                << "  theta_O " << ce[1] << " (surface frozen for "
                << nEq << " sub-steps)\n";
    }
    // The continuum re-solves the coverage of EVERY point at EVERY step, so a
    // point uncovered by etching is handed the steady-state coverage of its
    // local flux, not a bare surface. handDown is the cell equivalent: the
    // receding front passes its adsorbate to the cell it uncovers.
    if (std::getenv("PMC_HANDDOWN")) pmc.setHandDown(true);
    // Sidewall vs floor NEUTRAL ARRIVAL, the direct analogue of the level
    // set's O_flux field. Same geometric bands as the level-set analysis, so
    // the two ratios are comparable without fitting anything.
    auto arrivalReport = [&](const char *when) {
      if (!std::getenv("PMC_HITTALLY")) return;
      std::cout << "    [" << when << "]\n";
      const auto &hO = pmc.neutralHitsO();
      const auto &hF = pmc.neutralHitsF();
      const auto &hI = pmc.ionHitsPerCell();
      const auto &hY = pmc.ionYieldPerCell();
      const auto &hN = pmc.ionNzPerCell();
      const auto &dd = lat.dims();
      double wO = 0, wF = 0, fO = 0, fF = 0, wI = 0, fI = 0, wY = 0, fY = 0, wN = 0, fN = 0;
      size_t nw = 0, nf = 0, wStF = 0, wStO = 0;
      // The floor is the LOWEST COLUMN TOP inside the opening. Taking the
      // lowest solid cell instead lands on the bottom of the domain.
      T zfloor = T(0);
      std::array<int, D> idx{};
      for (int i = 1; i < dd[0] - 1; ++i) {
        const T xx = lat.minCorner()[0] + DX * (i + T(0.5));
        if (std::abs(xx) >= W / 2 - T(1.5)) continue;
        T top = -std::numeric_limits<T>::max();
        for (int j = 1; j < dd[1] - 1; ++j) {
          idx[0] = i; idx[1] = j; const int c = lat.cellId(idx);
          if (c < 0 || fill[c] < T(0.5)) continue;
          if (material[c] == (int)ps::Material::Mask) continue;
          const T zz = lat.minCorner()[1] + DX * (j + T(0.5));
          if (zz > top) top = zz;
        }
        if (top > -std::numeric_limits<T>::max() && top < zfloor) zfloor = top;
      }
      for (int j = 1; j < dd[1] - 1; ++j)          // skip the domain frame:
        for (int i = 1; i < dd[0] - 1; ++i) {      // its cells read as exposed
          idx[0] = i; idx[1] = j; const int c = lat.cellId(idx);
          if (c < 0 || fill[c] < T(0.5)) continue;
          if (material[c] == (int)ps::Material::Mask) continue;
          bool exposed = false;
          for (int d = 0; d < D && !exposed; ++d)
            for (int sg = -1; sg <= 1 && !exposed; sg += 2) {
              auto nb = idx; nb[d] += sg; const int b = lat.cellId(nb);
              if (b < 0 || fill[b] < T(0.5)) exposed = true;
            }
          if (!exposed) continue;
          const T xx = lat.minCorner()[0] + DX * (i + T(0.5));
          const T zz = lat.minCorner()[1] + DX * (j + T(0.5));
          const bool isWall = std::abs(std::abs(xx) - W / 2) < T(1.2) &&
                              zz < T(-0.3) && zz > zfloor + T(0.3);
          const bool isFloor = std::abs(xx) < W / 2 - T(1.5) &&
                               zz < T(-0.3) && zz > zfloor - T(0.4);
          const double hi = (c < (int)hI.size()) ? hI[c] : 0.0;
          const double hy = (c < (int)hY.size()) ? hY[c] : 0.0;
          const double hn = (c < (int)hN.size()) ? hN[c] : 0.0;
          if (isWall)  { wO += hO[c]; wF += hF[c]; wI += hi; wY += hy; wN += hn; ++nw;
                         if (pmc.states()[c] == 1) ++wStF;
                         else if (pmc.states()[c] == 2) ++wStO; }
          if (isFloor) { fO += hO[c]; fF += hF[c]; fI += hi; fY += hy; fN += hn; ++nf; }
        }
      const double BST = (double)pmc.tallyBoost();
      double totO = 0, totF = 0;
      for (size_t q = 0; q < hO.size(); ++q) { totO += hO[q]; totF += hF[q]; }
      std::cout << std::setprecision(6)
                << "    tally check: total O arrivals " << totO << " vs adsorbed "
                << pmc.nAdsO << ";  total F arrivals " << totF << " vs adsorbed "
                << pmc.nAdsF << "   (arrivals MUST exceed adsorptions)\n";
      std::cout << std::setprecision(4)
                << "    NEUTRAL ARRIVAL per exposed cell (floor z = " << zfloor << ")\n"
                << "      O: floor " << (nf ? fO / nf / BST : 0) << " (" << nf
                << " cells),  wall " << (nw ? wO / nw / BST : 0) << " (" << nw
                << " cells),  wall/floor " << ((nf && fO) ? (wO / nw) / (fO / nf) : 0)
                << "   [level set O_flux ratio 0.81]\n"
                << "      F: floor " << (nf ? fF / nf / BST : 0) << ",  wall "
                << (nw ? wF / nw / BST : 0) << ",  wall/floor "
                << ((nf && fF) ? (wF / nw) / (fF / nf) : 0)
                << "   [level set F_flux ratio 0.86]\n"
                << "      ION: floor " << (nf ? fI / nf / BST : 0) << ",  wall "
                << (nw ? wI / nw / BST : 0) << ",  wall/floor "
                << ((nf && fI) ? (wI / nw) / (fI / nf) : 0)
                << "   [raw hit count -- NOT comparable to a yield flux]\n"
                << "      ION YIELD-WEIGHTED: floor " << (nf ? fY / nf / BST : 0)
                << ",  wall " << (nw ? wY / nw / BST : 0) << ",  wall/floor "
                << ((nf && fY) ? (wY / nw) / (fY / nf) : 0)
                << "   [level set R4_yieldFlux ratio 0.0562]\n"
                << "      mean yield per wall hit " << (wI ? wY / wI : 0)
                << " vs per floor hit " << (fI ? fY / fI : 0)
                << "   (grazing on a vertical wall should be << floor)\n"
                << "      WALL PROFILE, per exposed wall cell, by depth:\n"
                << "        z[nm]   ionHits   O      F     n\n";
      {
        for (T zlo = T(0); zlo > zfloor; zlo -= T(0.5)) {
          double si = 0, so = 0, sf = 0; size_t n = 0;
          for (int j = 1; j < dd[1] - 1; ++j)
            for (int i = 1; i < dd[0] - 1; ++i) {
              std::array<int, D> q{i, j}; const int c = lat.cellId(q);
              if (c < 0 || fill[c] < T(0.5)) continue;
              if (material[c] == (int)ps::Material::Mask) continue;
              bool ex = false;
              for (int d = 0; d < D && !ex; ++d)
                for (int sg = -1; sg <= 1 && !ex; sg += 2) {
                  auto nb = q; nb[d] += sg; const int b = lat.cellId(nb);
                  if (b < 0 || fill[b] < T(0.5)) ex = true;
                }
              if (!ex) continue;
              const T xx = lat.minCorner()[0] + DX * (i + T(0.5));
              const T zz = lat.minCorner()[1] + DX * (j + T(0.5));
              if (std::abs(std::abs(xx) - W / 2) >= T(1.2)) continue;
              if (zz > zlo || zz <= zlo - T(0.5)) continue;
              si += (c < (int)hI.size()) ? hI[c] : 0.0;
              so += hO[c]; sf += hF[c]; ++n;
            }
          if (n)
            std::cout << "      " << std::setw(6) << std::setprecision(3) << zlo
                      << std::setw(10) << si / n << std::setw(8) << so / n
                      << std::setw(8) << sf / n << std::setw(6) << n << "\n";
        }
        std::cout << std::setprecision(4);
      }
      std::cout << "      WALL COVERAGE, SAME cells: theta_F "
                << (nw ? double(wStF) / nw : 0.0) << ",  theta_O "
                << (nw ? double(wStO) / nw : 0.0) << "\n";
      std::cout << "      mean |n_z| the impact USED: wall " << (wI ? wN / wI : 0)
                << ", floor " << (fI ? fN / fI : 0)
                << "   (a vertical wall must give ~0)\n";
    };
    const int tallyLast = std::getenv("PMC_TALLYLAST")
                              ? std::atoi(std::getenv("PMC_TALLYLAST")) : 0;
    for (int s = 0; s < steps; ++s) {
      // Reset the tally N steps from the end so arrivals and coverages
      // describe the SAME state: etch running, geometry barely moved.
      if (tallyLast > 0 && s == steps - tallyLast) pmc.resetHitTally();
      pmc.step(time / steps);
    }
    if (tallyLast > 0)
      arrivalReport("LAST STEPS OF THE ETCH: same state as the coverages");
    // Sidewall-restricted coverage. The continuum's theta_O = 0.993 is a
    // SIDEWALL number; pmc.coverages() averages the whole surface, floor
    // included, where ions clear O and both arms agree on a low value. The
    // two are not comparable, so measure the wall on its own: exposed,
    // non-mask solid cells out near the wall and below the mask foot.
    auto wallTheta = [&](T &tF, T &tO, size_t &n) {
      const auto &st = pmc.states();
      const auto &dd = lat.dims();
      size_t nF = 0, nO = 0; n = 0;
      std::array<int, D> idx{};
      for (int j = 0; j < dd[1]; ++j) {
        const T zz = lat.minCorner()[1] + DX * (j + T(0.5));
        if (zz > -T(0.3) || zz < -TARGET) continue;      // below the mask foot
        for (int i = 0; i < dd[0]; ++i) {
          const T xx = lat.minCorner()[0] + DX * (i + T(0.5));
          if (std::abs(xx) < W / 2 - T(1.5)) continue;   // wall band only
          idx[0] = i; idx[1] = j;
          const int c = lat.cellId(idx);
          if (c < 0 || fill[c] < T(0.5)) continue;
          if (material[c] == (int)ps::Material::Mask) continue;
          // exposed = at least one face neighbour is not solid
          bool exposed = false;
          for (int d = 0; d < D && !exposed; ++d)
            for (int sg = -1; sg <= 1 && !exposed; sg += 2) {
              auto nb = idx; nb[d] += sg;
              const int b = lat.cellId(nb);
              if (b < 0 || fill[b] < T(0.5)) exposed = true;
            }
          if (!exposed) continue;
          ++n; if (st[c] == 1) ++nF; else if (st[c] == 2) ++nO;
        }
      }
      tF = n ? T(nF) / n : T(0); tO = n ? T(nO) / n : T(0);
    };
    if (fluxOnly) { pmc.setFreezeSurface(false);
                    arrivalReport("FLUX ONLY: perfect trench, frozen, so this IS the flux"); }
    else arrivalReport("WHOLE RUN: surface moving, so this measures EXPOSURE TIME, not flux");
    { T tF, tO; size_t n; wallTheta(tF, tO, n);
      std::cout << std::setprecision(4) << "    SIDEWALL after the etch: theta_F "
                << tF << "  theta_O " << tO << "  over " << n << " cells\n"; }
    // Post-etch freeze: hold the geometry and keep the chemistry running, to
    // see where the coverages go when surface renewal is switched off
    // ENTIRELY. If theta_O climbs to the continuum's 0.993 the PMC chemistry
    // is right and the two arms differ only by the QSSA; if it plateaus low,
    // the O balance itself is wrong.
    if (const char *e = std::getenv("PMC_POSTEQ")) {
      const int nEq = std::atoi(e);
      pmc.setFreezeSurface(true);
      // Zero the ion flux for the freeze ONLY. The trench is etched with ions
      // as usual; the relaxation then runs without them, so an oxygen loss
      // that is ion-driven separates from one that is not.
      // Tally arrivals over the FROZEN window only, where the geometry is
      // static and arrivals per cell are a flux.
      if (std::getenv("PMC_HITTALLY")) pmc.resetHitTally();
      // Diagnostic pass: trace BOOST times the physical flux and let nothing
      // react, so the flux estimate has 1/sqrt(BOOST) the noise. Every tally
      // is divided by the boost below, so the numbers stay physical.
      if (const char *e = std::getenv("PMC_TALLYBOOST")) {
        pmc.setTallyOnly(true, std::atof(e));
        std::cout << "    [diagnostic pass: " << e
                  << "x flux, nothing reacts]\n";
      }
      const bool noIon = std::getenv("PMC_POSTEQ_NOION") != nullptr;
      if (noIon) { pmc.setIonFlux(T(0)); std::cout << "    (ions off for the freeze)\n"; }
      const int every = std::max(1, nEq / 12);
      std::cout << "    post-etch freeze, dt = " << (time / steps) * 1e6
                << " us/sub-step\n"
                << "      t[ms]  theta_F  theta_O  wall_F  wall_O   O ads\n";
      for (int s = 0; s < nEq; ++s) {
        pmc.step(time / steps);
        if (((s + 1) % every) == 0) {
          const auto ce = pmc.coverages();
          T tF, tO; size_t n; wallTheta(tF, tO, n);
          std::cout << "    " << std::setw(7) << std::setprecision(4)
                    << (s + 1) * (time / steps) * 1e3 << std::setw(9) << ce[0]
                    << std::setw(9) << ce[1] << std::setw(8) << tF
                    << std::setw(8) << tO << std::setw(8) << pmc.nAdsO << "\n";
        }
      }
      pmc.setFreezeSurface(false);
      arrivalReport("FROZEN WINDOW: geometry static, so arrivals per cell ARE a flux");
      pmc.setTallyOnly(false);
    }
    // Islands are cleared once, after the run: pruning every step removes
    // each one the moment it forms and runs the etch ahead of itself.
    if (pmc.pruneIslands()) {
      const size_t cut = pmc.pruneUnsupported();
      std::cout << "    pruned " << cut << " unsupported cells\n";
    }
    dump("cmp_" + tag + "_final.vtu");
    const auto c = pmc.coverages();
    { long cf = 0, cfd = 0; double cr = 0; pmc.capStats(cf, cfd, cr);
      std::cout << "    curvature cap fired on " << cfd << " of " << cf
                << " fits (" << (cf ? 100.0 * cfd / cf : 0.0)
                << " %),  mean capped radius " << cr << "\n"; }
    std::cout << std::setprecision(4) << "    theta_F " << c[0] << "  theta_O "
              << c[1] << "   cells removed " << pmc.removedCells() << "\n"
              << "    ion hits " << pmc.nIons << " (on F: " << pmc.nIonsOnF
              << "\n    neutrals launched " << pmc.nLaunch << ", surface hits "
              << pmc.nHitN << ", acceptance tests " << pmc.nAcc
              << "\n    funnel: on mask " << pmc.nHitMask << ", occupied "
              << pmc.nHitOccupied << ", sticking fail " << pmc.nStickFail
              << "\n    LATERAL DISPLACEMENT of removals [cells], mean |dx|:"
              << "  thermal " << (pmc.nDxTh ? pmc.sumDxTh / pmc.nDxTh : 0.0)
              << " (n=" << pmc.nDxTh << ")"
              << ",  ion-enh " << (pmc.nDxIE ? pmc.sumDxIE / pmc.nDxIE : 0.0)
              << " (n=" << pmc.nDxIE << ")"
              << ",  sputter " << (pmc.nDxSp ? pmc.sumDxSp / pmc.nDxSp : 0.0)
              << " (n=" << pmc.nDxSp << ")"
              << "\n    O* FULL BUDGET  (adsorbed " << pmc.nAdsO << ")"
              << "\n      ion-cleared, DIRECT ion    " << pmc.nClearODirect
              << "\n      ion-cleared, REFLECTED ion " << pmc.nClearORefl
              << "\n      cell removed, thermal      " << pmc.nORemTh
              << "\n      cell removed, ion-enhanced " << pmc.nORemIE
              << "\n      cell removed, sputter      " << pmc.nORemSp
              << "\n      reset by a neighbour       " << pmc.nOLostReset
              << "\n      thermally desorbed         " << pmc.nThermO
              << "\n      SUM                        "
              << (pmc.nClearODirect + pmc.nClearORefl + pmc.nORemTh + pmc.nORemIE
                  + pmc.nORemSp + pmc.nOLostReset + pmc.nThermO)
              << "   (the rest is still on the surface)"
              << "\n    O ARRIVALS: found bare " << pmc.nOHitBare
              << ", found occupied " << pmc.nOHitOccupied
              << "   -> competition loss "
              << std::setprecision(3)
              << 100.0 * pmc.nOHitOccupied /
                     std::max<size_t>(pmc.nOHitBare + pmc.nOHitOccupied, 1)
              << " %" << std::setprecision(4)
              << "\n    O* DEATHS: ion-cleared " << pmc.nClearO
              << ", cell itself removed " << pmc.nOLostRemoved
              << ", reset when a neighbour went " << pmc.nOLostReset
              << ", thermally desorbed " << pmc.nThermO
              << "  (adsorbed " << pmc.nAdsO << ")"
              << "\n    O* ion-clearing location: on WALL cells " << pmc.nClearOWall
              << " (of those, impact also on a wall cell " << pmc.nClearOWallDirect
              << "), on floor cells " << pmc.nClearOFloor
              << "\n    O* budget: adsorbed " << pmc.nAdsO
              << ", ion-cleared " << pmc.nClearO
              << ", thermally desorbed " << pmc.nThermO
              << "\n    F* budget: adsorbed " << pmc.nAdsF
              << ", ion-cleared " << pmc.nClearF
              << ", thermal firings " << pmc.nThermF
              << ", bare resets " << pmc.nBareReset
              << "  -> created " << pmc.nAdsF << " vs destroyed "
              << (pmc.nClearF + pmc.nThermF)
              << ", site fail " << pmc.nSiteFail << ", adsorbed "
              << (pmc.nAdsF + pmc.nAdsO)
              << ", rejected " << pmc.nAccClamped << " ("
              << 100.0 * pmc.nAccClamped / std::max<size_t>(pmc.nAcc, 1)
              << " %), reflections " << pmc.nBounce
              << "\n    ion hits " << pmc.nIons << " (on F: " << pmc.nIonsOnF
              << "),  F ads " << pmc.nAdsF << ",  O ads " << pmc.nAdsO
              << ",  bare resets " << pmc.nBareReset
              << " (per adsorption "
              << double(pmc.nBareReset) /
                     std::max<size_t>(pmc.nAdsF + pmc.nAdsO, 1)
              << ")\n    thermal firings "
              << pmc.nThermF << "\n"
              << "    mean cos(theta) seen by ions "
              << (pmc.nCosIon ? pmc.sumCosIon / pmc.nCosIon : 0.0)
              << "   (1.0 = every impact reads normal incidence)\n"
              << "    cells removed by channel: ion-enhanced " << pmc.remIE
              << "  sputter " << pmc.remSp << "  thermal " << pmc.remTh
              << "   (continuum split 82.4 / 0.9 / 12.6 %)\n";
    if (tag == "pmciface") {
      std::cout << "    x[nm]  ionHits  onF  ieRem  thermRem  (trench -20..20)\n";
      for (size_t q = 0; q < pmc.histHit.size(); ++q) {
        const T xx = lat.minCorner()[0] + DX * (q + T(0.5));
        if (std::abs(xx) > 24 || ((int)(xx + 100)) % 2) continue;
        std::cout << "    " << std::setw(6) << std::setprecision(1) << xx
                  << std::setw(8) << pmc.histHit[q] << std::setw(6)
                  << pmc.histOnF[q] << std::setw(7) << pmc.histRem[q]
                  << std::setw(10) << (q < pmc.histTh.size() ? pmc.histTh[q] : 0)
                  << "\n";
      }
      {
        size_t rr = 0, dd = 0;
        for (size_t q = 0; q < pmc.reflRem.size(); ++q) { rr += pmc.reflRem[q]; dd += pmc.directRem[q]; }
        std::cout << "    ion-enhanced events: direct " << dd << ", REFLECTED " << rr
                  << "  (" << std::setprecision(3) << 100.0*rr/std::max<size_t>(rr+dd,1)
                  << " %)\n    reflected arrival angle:";
        for (int q = 0; q < 9; ++q)
          if (pmc.reflAng[q]) std::cout << "  " << 10*q << "-" << 10*q+10 << ":" << pmc.reflAng[q];
        {
          size_t tf = 0, tw = 0;
          for (int q = 0; q < 9; ++q) { tf += pmc.reflSrcFloor[q]; tw += pmc.reflSrcWall[q]; }
          std::cout << "\n    reflections BORN on floor cells " << tf
                    << ", on wall cells " << tw << "  (floor share "
                    << std::setprecision(3)
                    << 100.0 * tf / std::max<size_t>(tf + tw, 1) << " %)"
                    << "\n      a smooth floor under an exponent-500 beam is hit"
                       " at ~0 deg and can reflect NOTHING\n"
                    << "      incidence at birth, floor:";
          for (int q = 0; q < 9; ++q)
            if (pmc.reflSrcFloor[q]) std::cout << "  " << 10*q << "-" << 10*q+10 << ":" << pmc.reflSrcFloor[q];
          std::cout << "\n      incidence at birth, wall :";
          for (int q = 0; q < 9; ++q)
            if (pmc.reflSrcWall[q]) std::cout << "  " << 10*q << "-" << 10*q+10 << ":" << pmc.reflSrcWall[q];
          std::cout << std::setprecision(4);
        }
        std::cout << "\n    wall-cell ion hits " << pmc.nWallHits
                  << ", of which < 20 deg: " << pmc.nWallNear
                  << "  (" << std::setprecision(3)
                  << 100.0 * pmc.nWallNear / std::max<size_t>(pmc.nWallHits, 1)
                  << " %)\n      of those near-normal wall hits: estimator had"
                     " NO normal (face fallback) " << pmc.nWallNearFace
                  << ", entered through the TOP face " << pmc.nWallNearTop
                  << std::setprecision(4);
        if (!pmc.wallTrace.empty())
          std::cout << "\n    near-normal wall hits, in full:\n" << pmc.wallTrace;
        std::cout << "\n    reflections per ion:";
        for (int q = 0; q < 12; ++q)
          if (pmc.bounceHist[q]) std::cout << "  " << q << ":" << pmc.bounceHist[q];
        std::cout << "\n    x[nm]   direct  reflected   (trench -20..20)\n";
        for (size_t q = 0; q < pmc.reflRem.size(); ++q) {
          const T xx = lat.minCorner()[0] + DX * (q + T(0.5));
          if (std::abs(xx) > 24 || ((int)(xx + 100)) % 2) continue;
          std::cout << "    " << std::setw(6) << std::setprecision(1) << xx
                    << std::setw(9) << pmc.directRem[q] << std::setw(11) << pmc.reflRem[q] << "\n";
        }
        std::cout << std::setprecision(4);
      }
      std::cout << "    theta   floorHits  <Y_ie>    wallHits  <Y_ie>   f_ion\n";
      for (int q = 0; q < 9; ++q) {
        if (!pmc.floorHit[q] && !pmc.wallHit[q]) continue;
        const double th = 10.0 * q + 5.0;
        const double fi = th <= 60 ? 1.0 : std::max(3.0 - 6.0*th/180.0, 0.0);
        std::cout << "    " << std::setw(3) << (int)(10*q) << "-" << std::setw(3)
                  << (int)(10*q+10) << std::setw(10) << pmc.floorHit[q]
                  << std::setw(9) << std::setprecision(3)
                  << (pmc.floorHit[q] ? pmc.floorY[q]/pmc.floorHit[q] : 0.0)
                  << std::setw(11) << pmc.wallHit[q] << std::setw(9)
                  << (pmc.wallHit[q] ? pmc.wallY[q]/pmc.wallHit[q] : 0.0)
                  << std::setw(8) << fi << "\n";
      }
      std::cout << std::setprecision(4);
      if (false) std::cout << "";
      for (int q = 0; q < 9; ++q) {
        if (!pmc.angHit[q]) continue;
        const double th = 10.0 * q + 5.0;
        const double fion = th <= 60 ? 1.0 : std::max(3.0 - 6.0*th/180.0, 0.0);
        const double fa = pmc.angHit[q] / std::max(pmc.angArea[q], 1e-9);
        const double fa0 = pmc.angHit[0] / std::max(pmc.angArea[0], 1e-9);
        std::cout << "    " << std::setw(3) << (int)(10*q) << "-" << std::setw(3)
                  << (int)(10*q+10) << std::setw(7) << pmc.angHit[q]
                  << std::setw(10) << std::setprecision(4) << pmc.angArea[q]
                  << std::setw(11) << std::setprecision(3) << fa
                  << std::setw(9) << fa/fa0 << std::setw(9)
                  << std::cos(th*M_PI/180.0) << std::setw(8) << fion << "\n";
      }
      std::cout << std::setprecision(4);
    }
  }
}
