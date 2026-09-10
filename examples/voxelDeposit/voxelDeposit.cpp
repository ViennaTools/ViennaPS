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
//   DEP_PMCONLY   skip the two continuum arms
//   DEP_BOUNCE    re-emissions a molecule may make before it is discarded
//   PMC_FITR / PMC_REFLR / PMC_PWIN / PMC_MINPTS / PMC_NOPRUNE / PMC_NOPLANE
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

// the substrate has to reach below the trench and the gas well above the
// field, or the film grows out of the cell set
static T yExtent() { return 2 * (H + 40); }
static T floorY()  { return -H - 20; }

static ps::SmartPointer<ps::Domain<T, D>> makeDomain() {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, T(4) * W, yExtent());
  ps::MakeTrench<T, D>(dom, W, H, T(0), T(0), T(0), false, ps::Material::Si)
      .apply();
  return dom;
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
  const bool pmcOnly = std::getenv("DEP_PMCONLY") != nullptr;
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");
  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/conformal.mechanism.json");

  // DEP_S0 overrides the sticking of the one adsorption step. It appears in
  // the rate law AND in the particle's re-emission, so both have to move.
  if (const char *e = std::getenv("DEP_S0")) {
    const T s0 = std::atof(e);
    for (auto &g : mech.gas)
      if (g.traced && !g.isIonChannel) {
        g.s0 = s0;
        g.stickingConstant = ps::MaterialValueMap<
            typename ps::ChemicalMechanism<T>::RateConstant>::fromDefault(
            typename ps::ChemicalMechanism<T>::RateConstant{s0, 0, 0});
      }
    for (auto &r : mech.reactions)
      if (r.isAdsorption)
        r.prefactor = s0;
  }

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
            << "trench W=" << W << " depth=" << H << " dx=" << DX
            << ",  growth rate " << GR << " nm/s,  t = " << time << " s ("
            << TARGET << " nm on the open field)\n"
            << "PMC sticking probability per impact " << pStick << "\n\n";

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
    proc.setParameters(cov);
    std::cout << "level set:\n";
    proc.apply();
    writeSurface(dom, "dep_ls_final.vtp");
  }

  // --------------------------------------------------- filling-fraction voxel
  if (!pmcOnly) {
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
    pmc.setDeposition(true);
    pmc.setDepositionP(pStick);
    if (const char *e = std::getenv("DEP_BOUNCE")) pmc.setMaxBounce(std::atoi(e));
    if (const char *e = std::getenv("PMC_FITR")) pmc.setFitRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_REFLR")) pmc.setReflectRadius(std::atoi(e));
    if (const char *e = std::getenv("PMC_PWIN")) pmc.setPlaneWindow(std::atof(e));
    if (const char *e = std::getenv("PMC_MINPTS")) pmc.setMinFitPoints(std::atoi(e));
    if (std::getenv("PMC_NOPLANE")) pmc.setPlaneAcceptance(false);
    if (std::getenv("PMC_NOPRUNE")) pmc.setPruneIslands(false);
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
    std::cout << "binary-cell PMC:\n";
    dump("dep_pmc_initial.vtu");
    const int steps = 300;
    for (int s = 0; s < steps; ++s)
      pmc.step(time / steps);
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
