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
  ps::MakeTrench<T, D>(dom, W, T(0), T(0), MASKH, T(0), false,
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
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
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
    ps::Process<T, D> proc(dom, model, time);
    proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
    ps::RayTracingParameters rt;
    rt.raysPerPoint = 400; rt.useRandomSeeds = false; rt.rngSeed = 1000;
    proc.setParameters(rt);
    ps::CoverageParameters cov; cov.tolerance = 1e-6; cov.maxIterations = 40;
    proc.setParameters(cov);
    std::cout << "level set:\n";
    proc.apply();
    writeSurface(dom, "cmp_ls_final.vtp");
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
    cells->addScalarData("State", 0.);   // before any reference is taken
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
    if (const char *e = std::getenv("PMC_P")) pmc.setSimpleP(std::atof(e));
    if (const char *e = std::getenv("PMC_BOUNCE")) pmc.setMaxBounce(std::atoi(e));
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
    for (int s = 0; s < steps; ++s)
      pmc.step(time / steps);
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
              << "),  F adsorptions " << pmc.nAdsF << ",  thermal firings "
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
