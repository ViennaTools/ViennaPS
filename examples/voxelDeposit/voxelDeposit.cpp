// Conformal deposition into a trench, written out by all three surface
// representations so they can be laid over one another:
//
//   level set        an implicit surface, read as a mesh          -> .vtp/.csv
//   voxel            cells carrying a FILLING FRACTION in [0,1]   -> .vtu
//   binary-cell PMC  cells that are solid or gas                  -> .vtu
//
// The chemistry is one reaction with one decision (reactions/conformal.yaml):
//
//     A -> Si,  adsorption, sticking s0
//
// A precursor that hits the surface sticks with probability s0 and becomes
// solid there; otherwise it re-emits diffusively and carries on into the
// trench. No coverage, no thermal channel, no ion, so nothing here depends on
// the angle of incidence and the three arms differ only in transport.
//
// s0 is the conformality knob: at s0 = 1 the film is shadow limited and pinches
// off at the mouth, and as s0 falls a molecule gets many more chances further
// down, so the step coverage climbs toward one. DEP_S0 sets it; the growth rate
// and hence the process time follow from it automatically.
//
//   ./voxelDeposit [thickness_nm] [seed] [dx]
//
//   DEP_S0=0.02   sticking probability, overriding the mechanism's
//   DEP_W, DEP_H  trench width and depth in nm
//   DEP_BLANKET=1 a flat wafer 4*DEP_W wide instead of the trench
//   DEP_PMCONLY   skip the two continuum arms
//   DEP_FF=1      run the filling-fraction arm as well (off by default)
//   DEP_BOUNCE    re-emissions a molecule may make before it is discarded
//   MECH_FILE     the mechanism, default reactions/conformal.mechanism.json
//   LS_PREEQ=1    pre-equilibrate the level set's coverages (comparison only;
//                 by default it starts bare, like the cell set)
//   PMC_FITR / PMC_REFLR / PMC_PWIN / PMC_MINPTS / PMC_PRUNE / PMC_PLANE
//
// SILANE CVD. With MECH_FILE=.../silane.mechanism.json the cell set runs the
// mechanism's own surface kinetics (VoxelPMC::setCVD): SiH4 adsorbs on a free
// site pair with s(T), SiH3* grows the film at k1, H* pairs leave as H2 at k2,
// all three read from the mechanism at its temperature. The time step keeps
// the R3 probability per step at 0.025 or below.
//   PMC_CVDMIX    H* hops per R3 attempt (default 100)
//   PMC_CVDRELAX  reach of the R2 growth site in cells (default 4)
//   DEP_STEPS     cell-set steps, overriding that choice
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeTrench.hpp>
#include <process/psProcess.hpp>

#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <string>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 2;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

static T DX = 1.0;                  // argv[3]: grid spacing
static T W = 40.0, H = 80.0;        // trench width and depth, nm
static T TARGET = 12.0;             // argv[1]: nm of film on the open field
static unsigned SEED = 7;           // argv[2]
static bool BLANKET = false;        // DEP_BLANKET

// the substrate has to reach below the trench and the gas well above the
// field, or the film grows out of the cell set
static T yExtent() { return 2 * (H + 40); }
static T floorY()  { return BLANKET ? T(-20) : -H - 20; }

static ps::SmartPointer<ps::Domain<T, D>> makeDomain() {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, T(4) * W, yExtent());
  if (BLANKET)
    ps::MakePlane<T, D>(dom, T(0), ps::Material::Si).apply();
  else
    ps::MakeTrench<T, D>(dom, W, H, T(0), T(0), T(0), false, ps::Material::Si)
        .apply();
  return dom;
}

// mean height of the level-set surface, for the blanket
static T meanHeight(ps::SmartPointer<ps::Domain<T, D>> dom) {
  auto mesh = ps::SmartPointer<ls::Mesh<T>>::New();
  ls::ToSurfaceMesh<T, D>(dom->getLevelSets().back(), mesh).apply();
  T s = 0;
  for (const auto &n : mesh->getNodes()) s += n[D - 1];
  return mesh->getNodes().empty() ? T(0) : s / T(mesh->getNodes().size());
}

static ps::SmartPointer<cs::DenseCellSet<T, D>>
makeCells(ps::SmartPointer<ps::Domain<T, D>> dom) {
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., floorY()}, n[D] = {0., 1.};
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
  cs_->fromLevelSets(lss, mm, TARGET + T(8));
  return cs_;
}

static void writeSurface(ps::SmartPointer<ps::Domain<T, D>> dom,
                         const std::string &name) {
  auto mesh = ps::SmartPointer<ls::Mesh<T>>::New();
  ls::ToSurfaceMesh<T, D>(dom->getLevelSets().back(), mesh).apply();
  ls::VTKWriter<T>(mesh, name).apply();
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
  if (argc > 1) TARGET = std::atof(argv[1]);
  if (argc > 2) SEED = (unsigned)std::atoi(argv[2]);
  if (argc > 3) DX = std::atof(argv[3]);
  if (const char *e = std::getenv("DEP_W")) W = std::atof(e);
  if (const char *e = std::getenv("DEP_H")) H = std::atof(e);
  BLANKET = std::getenv("DEP_BLANKET") != nullptr;
  const bool pmcOnly = std::getenv("DEP_PMCONLY") != nullptr;
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");
  const std::string mechFile =
      std::getenv("MECH_FILE")
          ? std::string(std::getenv("MECH_FILE"))
          : std::string(VIENNAPS_MECHANISM_DIR) + "/conformal.mechanism.json";
  auto mech = ps::readChemicalMechanism<T>(mechFile);
  std::cout << "mechanism " << mechFile << "\n";

  // DEP_S0 overrides the sticking of the one adsorption step, as the sticking
  // at the mechanism's temperature (no activation energy). It appears in the
  // rate law AND in the particle's re-emission, so both have to move. The
  // rate law reads materialConstant through rateConstantsFor, and the growth
  // rate below, hence the process time and the cell set's sticking, is
  // computed through it; setting the prefactor alone left them at the file's
  // value while the level set took the override.
  if (const char *e = std::getenv("DEP_S0")) {
    const T s0 = std::atof(e);
    using RC = typename ps::ChemicalMechanism<T>::RateConstant;
    for (auto &g : mech.gas)
      if (g.traced && !g.isIonChannel) {
        g.s0 = s0;
        g.stickingEa = 0;
        g.stickingBeta = 0;
        g.stickingConstant =
            ps::MaterialValueMap<RC>::fromDefault(RC{s0, 0, 0});
      }
    for (auto &r : mech.reactions)
      if (r.isAdsorption) {
        r.prefactor = s0;
        r.Ea = 0;
        r.beta = 0;
        r.materialConstant = ps::MaterialValueMap<RC>::fromDefault(RC{s0, 0, 0});
      }
  }
  // the silane mechanism's two adsorbates: the cell set runs its kinetics
  const bool cvd = mech.coverageNames.size() == 2 &&
                   mech.coverageNames[0] == "SiH3*" &&
                   mech.coverageNames[1] == "H*" && mech.reactions.size() == 3;

  const auto gam = mech.sourceFluxes(ps::Material::Si);
  const auto kc = mech.rateConstantsFor(ps::Material::Si);
  std::vector<T> th(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gam, kc, th);
  const T GR = mech.growthRate(gam, kc, th, ps::Material::Si);
  if (GR <= T(0)) {
    std::cout << "the mechanism does not grow (rate " << GR << ")\n";
    return 1;
  }
  const T time = TARGET / GR;

  // the PMC's own knob: the probability that one impact sticks. On a blanket
  // the film grows at flux*sigma0*p/rho, so p is fixed by the same rate the
  // other two arms use.
  const T sigma0 = 10.0, rho = mech.solids.front().rho * 10.0; // 1e22/cm3 -> /nm3
  T fluxA = 0;
  for (const auto &g : mech.gas)
    if (g.traced && !g.isIonChannel) fluxA = g.sourceFlux;
  const T pStick = GR * rho / (fluxA * sigma0);

  std::cout << std::fixed << std::setprecision(4)
            << (BLANKET ? "blanket, width " : "trench W=")
            << (BLANKET ? T(4) * W : W);
  if (!BLANKET) std::cout << " depth=" << H;
  std::cout << " dx=" << DX << ",  growth rate " << GR << " nm/s,  t = "
            << time << " s (" << TARGET << " nm on the open field)\n";
  if (cvd) {
    std::cout << std::scientific << std::setprecision(4) << "silane CVD at "
              << mech.temperature << " K, flux " << fluxA << ":\n";
    for (size_t r = 0; r < mech.reactions.size(); ++r)
      std::cout << "    " << mech.reactions[r].equation << "   "
                << (r == 0 ? "s = " : "k = ") << kc[r] << "\n";
    std::cout << std::fixed << std::setprecision(5)
              << "    rate law on a flat wafer: theta_SiH3 " << th[0]
              << ",  theta_H " << th[1] << ",  free "
              << 1 - th[0] - th[1] << "\n\n";
  } else {
    std::cout << "PMC sticking probability per impact " << pStick << "\n\n";
  }

  // ------------------------------------------------------------- level set
  if (!pmcOnly) {
    auto dom = makeDomain();
    writeSurface(dom, "dep_ls_initial.vtp");
    auto model = ps::SmartPointer<ps::SurfaceChemistry<T, D>>::New(mech);
    ps::Process<T, D> proc(dom, model, time);
    proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
    ps::RayTracingParameters rt;
    rt.raysPerPoint = 400; rt.useRandomSeeds = false; rt.rngSeed = 1000;
    proc.setParameters(rt);
    ps::CoverageParameters cov; cov.tolerance = 1e-6; cov.maxIterations = 40;
    // NO PRE-EQUILIBRATION IN EITHER ARM: the level set starts from a bare
    // wafer, like the cell set. LS_PREEQ=1 restores the initialisation, for a
    // comparison only.
    if (!std::getenv("LS_PREEQ")) cov.maxIterations = 0;
    proc.setParameters(cov);
    std::cout << "level set"
              << (std::getenv("LS_PREEQ") ? " (coverages pre-equilibrated)"
                                          : " (starts bare)")
              << ":\n";
    proc.apply();
    writeSurface(dom, "dep_ls_final.vtp");
    if (BLANKET)
      std::cout << "    film " << meanHeight(dom) << " nm against the rate "
                << "law's " << GR * time << " nm\n";
  }

  // --------------------------------------------------- filling-fraction voxel
  // Off unless DEP_FF=1: the comparison is the level set against the cell set.
  if (!pmcOnly && std::getenv("DEP_FF")) {
    auto cells = makeCells(makeDomain());
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> fill(cells->getNumberOfCells(), T(0));
    std::vector<int> material(cells->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    const std::vector<T> fill0 = fill;
    cells->addScalarData("State", 0.);
    ps::VoxelChemistry<T, D> vox(mech, lat, fill, material);
    vox.setRaysPerCell(400);
    vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
    if (const char *e = std::getenv("DEP_SPREAD"))
      vox.setSurplusSpreading(std::atoi(e));
    if (const char *e = std::getenv("DEP_UNIFORMV"))
      vox.setUniformVelocity(std::atof(e) != 0 ? std::atof(e) * GR : GR);
    if (const char *e = std::getenv("DEP_NGATE"))
      vox.setFacetGate(std::atof(e));
    if (const char *e = std::getenv("DEP_MINAREA"))
      vox.setMinimumArea(std::atof(e) * std::pow(DX, D - 1));
    std::cout << "    surplus spreading " << vox.surplusSpreading()
              << ",  facet gate " << vox.facetGate()
              << ",  minimum interface area " << vox.minimumArea()
              << " (face " << std::pow(DX, D - 1) << ")\n";
    auto cov = vox.makeCoverages();
    vox.initialiseCoverages(cov, 1000u, 100, T(1e-6));
    cells->addScalarData("Velocity", 0.);
    cells->addScalarData("Flux", 0.);
    cells->addScalarData("Area", 0.);
    auto dump = [&](const std::string &name) {
      auto &ff = *cells->getFillingFractions();
      auto &mmv = *cells->getScalarData("Material");
      auto &state = *cells->getScalarData("State");
      auto &vv = *cells->getScalarData("Velocity");
      auto &fx = *cells->getScalarData("Flux");
      auto &ar = *cells->getScalarData("Area");
      for (size_t c = 0; c < fill.size(); ++c) {
        vv[c] = vox.lastVelocity().empty() ? 0. : vox.lastVelocity()[c];
        fx[c] = vox.lastFlux().empty() ? 0. : vox.lastFlux()[c];
        ar[c] = vox.lastArea().empty() ? 0. : vox.lastArea()[c];
      }
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        // ANY cell holding material is written. refreshMaterials only calls a
        // cell solid above fill 0.5, so taking its label here drops the whole
        // partial front -- which is the part a deposition profile is made of.
        mmv[c] = fill[c] <= T(1e-6)
                     ? T((int)ps::Material::GAS)
                     : T(material[c] == (int)ps::Material::GAS
                             ? (int)ps::Material::Si
                             : material[c]);
        // 0 substrate, 1 film -- a cell that held no material at t=0
        state[c] = (fill0[c] <= T(0.5) && fill[c] > T(1e-6)) ? T(1) : T(0);
      }
      cells->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };
    std::cout << "filling-fraction voxel:\n";
    dump("dep_voxel_initial.vtu");
    const int steps = 200;
    const int every = std::getenv("DEP_VOXEVERY")
                          ? std::atoi(std::getenv("DEP_VOXEVERY")) : 0;
    for (int s = 0; s < steps; ++s) {
      const auto r = vox.step(time / steps, cov, 100003u + s);
      if (every && (s + 1) % every == 0) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "dep_voxel_s%04d.vtu", s + 1);
        dump(nm);
      }
      if (s == 0 || s == steps - 1)
        std::cout << "    step " << s << ": " << r.surfaceCells
                  << " surface cells, v mean " << r.meanVelocity << " ["
                  << r.minVelocity << ", " << r.maxVelocity
                  << "] nm/s,  volume moved " << r.volumeMoved << " lost "
                  << r.volumeLost << "\n";
    }
    dump("dep_voxel_final.vtu");
  }

  // ------------------------------------------------------- binary-cell PMC
  {
    auto cells = makeCells(makeDomain());
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> fill(cells->getNumberOfCells(), T(0));
    std::vector<int> material(cells->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    const std::vector<T> fill0 = fill;
    cells->addScalarData("State", 0.);
    ps::VoxelPMC<T, D>::Parameters p;
    p.fluxF = fluxA;      // the one traced species rides the neutral channel
    p.fluxO = 0; p.fluxIon = 0;
    p.rho = rho;
    p.sigma0 = sigma0;
    ps::VoxelPMC<T, D> pmc(lat, fill, material, p);
    pmc.setSeed(SEED);
    const T nu = sigma0 * std::pow(DX, D - 1);   // sites per cell
    int steps = 300;
    if (cvd) {
      // the estimator the SF6/O2 comparison runs on (voxelCompare, pmciface)
      pmc.setNormalEstimator(cs::NormalEstimator::InterfaceAverage);
      pmc.setCVD(true, kc[0], kc[1], kc[2]);
      if (const char *e = std::getenv("PMC_CVDMIX"))
        pmc.setCVDMixing(std::atof(e));
      if (const char *e = std::getenv("PMC_CVDRELAX"))
        pmc.setCVDRelaxRadius(std::atoi(e));
      // A cell fires R3 with probability nu*k2*dt per step; keep it <= 0.025.
      // The arrivals of a step all come before its timed events, and at 0.05
      // that splitting held theta_H 2.6 % under the rate law on the s = 0.05
      // blanket (0.505 against 0.519), at 0.025 0.6 % (0.516).
      steps = std::max(steps, (int)std::ceil(time * nu * kc[2] / 0.025));
    } else {
      pmc.setDeposition(true);
      pmc.setDepositionP(pStick);
    }
    if (const char *e = std::getenv("DEP_STEPS")) steps = std::atoi(e);
    if (const char *e = std::getenv("DEP_BOUNCE")) pmc.setMaxBounce(std::atoi(e));
    if (const char *e = std::getenv("PMC_FITR")) pmc.setFitRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_REFLR")) pmc.setReflectRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_PWIN")) pmc.setPlaneWindow(std::atof(e));
    if (const char *e = std::getenv("PMC_MINPTS")) pmc.setMinFitPoints(std::atoi(e));
    if (std::getenv("PMC_PLANE")) pmc.setPlaneAcceptance(true);
    if (std::getenv("PMC_PRUNE")) pmc.setPruneIslands(true);
    auto dump = [&](const std::string &name) {
      auto &ff = *cells->getFillingFractions();
      auto &mmv = *cells->getScalarData("Material");
      auto &state = *cells->getScalarData("State");
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        const bool solid = fill[c] >= T(0.5);
        mmv[c] = solid ? T(material[c]) : T((int)ps::Material::GAS);
        state[c] = (solid && fill0[c] <= T(0.5)) ? T(1) : T(0);
      }
      cells->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };
    std::cout << "binary-cell PMC" << (cvd ? " (silane CVD, H* hops per R3 "
                                             "attempt " : "");
    if (cvd) std::cout << pmc.cvdMixing() << ", growth reach "
                       << pmc.cvdRelaxRadius() << " cells)";
    std::cout << ", " << steps << " steps of " << std::scientific
              << std::setprecision(3) << time / steps << " s:\n"
              << std::fixed;
    dump("dep_pmc_initial.vtu");
    // coverages, averaged over the second half of the run
    std::array<double, 3> census{0, 0, 0};
    int nCensus = 0;
    const int every = std::max(1, steps / 400);
    size_t cellsHalf = 0;   // grown cells at the half-way point
    for (int s = 0; s < steps; ++s) {
      if (2 * s == steps || 2 * s == steps + 1) cellsHalf = pmc.depositedCells();
      pmc.step(time / steps);
      if (cvd && 2 * s >= steps && (s % every) == 0) {
        const auto c = pmc.cvdCensus();
        for (int k = 0; k < 3; ++k) census[k] += double(c[k]);
        ++nCensus;
      }
    }
    if (cvd) {
      const double tot = census[0] + census[1] + census[2];
      std::cout << std::setprecision(5)
                << "    coverages, second half (" << nCensus << " samples of "
                << (nCensus ? tot / nCensus : 0.) << " surface cells): "
                << "SiH3 " << census[1] / std::max(tot, 1.) << ",  H "
                << census[2] / std::max(tot, 1.) << ",  free "
                << census[0] / std::max(tot, 1.) << "\n"
                << "    R1 adsorbed " << pmc.nCvdAds << ";  after the sticking "
                << "draw: site taken " << pmc.nCvdSiteBusy << ", neighbour "
                << "taken " << pmc.nCvdMateBusy << ", no neighbour "
                << pmc.nCvdNoMate << "\n"
                << "    R2 " << pmc.nCvdGrow << " (cells grown "
                << pmc.nCvdGrowCell << "),  R3 pairs " << pmc.nCvdH2
                << " (firings without a partner " << pmc.nCvdR3Miss << ")\n"
                << "    H* hops " << pmc.nCvdHop << " (blocked "
                << pmc.nCvdHopBlocked << "),  adsorbates moved off a grown "
                << "cell " << pmc.nCvdMoved << ", lost " << pmc.nCvdLost << "\n";
      if (BLANKET) {
        // the second half leaves out the bare-start transient
        const double perNm = std::pow(DX, D) / (T(4) * W);
        const double tHalf = time * double(steps - (steps + 1) / 2) / steps;
        std::cout << "    film " << double(pmc.depositedCells()) * perNm
                  << " nm against the rate law's " << GR * time << " nm;"
                  << "  second half " << double(pmc.depositedCells() - cellsHalf) *
                                             perNm / tHalf
                  << " nm/s against " << GR << "\n";
      }
    }
    if (pmc.pruneIslands())
      std::cout << "    pruned " << pmc.pruneUnsupported() << " unsupported cells\n";
    dump("dep_pmc_final.vtu");
    const double perCell = p.rho * std::pow(DX, D);
    std::cout << std::setprecision(4)
              << "    launched " << pmc.nLaunch << ",  surface hits "
              << pmc.nHitN << "  (" << (double)pmc.nHitN / std::max<size_t>(pmc.nLaunch, 1)
              << " per particle)\n"
              << "    stuck " << pmc.nDepStick << " (" << pmc.nDepFail
              << " with nowhere to go),  cells grown " << pmc.depositedCells()
              << "  (dose " << pmc.nDepStick / perCell << " cells)\n"
              << "    reflections " << pmc.nBounce << ",  escaped " << pmc.nEscape
              << ",  capped " << pmc.nBounceCap << "\n";
  }
}
