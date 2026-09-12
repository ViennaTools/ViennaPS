// The C2F6/SiO2 mechanism of Zhang & Kushner (2001) on a masked SiO2 trench,
// written out by the level set and by the binary-cell PMC on the same
// geometry.
//
// Both arms read reactions/c2f6_sio2.mechanism.json for the geometry, the etch
// time and the rate; the PMC's own copy of Table I is in psVoxelPMC.hpp, and
// the three features that file cannot express -- lambda([P]), Eq. (4), and the
// bulk/top-layer distinction -- are named in its header.
//
// WHY A SEPARATE HARNESS. The shared surfaceChemistry example builds its
// substrate as Material::Si, and this mechanism gates every passivation step
// to SiO2, so on that geometry the SiF_x ladder is never seeded and the etch
// runs on one channel. The substrate material has to be the one the chemistry
// is written for.
//
//   ./voxelFCTrench [target_nm] [seed] [dx_nm]
//
//   FCT_W, FCT_MASK   opening and mask height in nm
//   FCT_LSONLY / FCT_PMCONLY
//   FCT_SCALE   multiply every PMC probability by k and divide its time by k.
//               The trajectory is invariant under this -- the coverages, the
//               polymer thickness and the profile at a given depth are all
//               unchanged, only the clock moves -- so it buys wall time at the
//               cost of sqrt(k) more shot noise per unit depth. The level set
//               is cheap and always runs unscaled, so the two arms still meet
//               at the same depth.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <psUnits.hpp>
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
#include <string>
#include <vector>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 2;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

static T DX = 0.5;
static T W = 40.0, MASKH = 30.0;
// From flat ground, taper needs far more etch than is affordable before the
// feature is deep enough for the sidewalls to matter at all -- 2.9 nm into a
// 40 nm opening is flat ground. Start deep instead, and watch what the etch
// does to the profile it inherits.
static T DEPTH = 0.0;            // initial trench depth, nm
static T TAPER = 0.0;            // initial sidewall taper, degrees
// Substrate to leave below the feature. In the paper the taper is an OUTCOME
// of a long etch through a thick resist, not a starting shape, so the run has
// to have somewhere to go: this sizes the domain for the depth expected, not
// for the depth it starts at.
static T ROOM = 40.0;
static T TARGET = 20.0;          // nm of blanket-equivalent etch
static unsigned SEED = 7;

static ps::SmartPointer<ps::Domain<T, D>> makeDomain(T dx) {
  const T vertical = T(2) * (DEPTH + MASKH + ROOM);
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(dx, T(3) * W, vertical);
  ps::MakeTrench<T, D>(dom, W, DEPTH, TAPER, MASKH, T(0), false,
                       ps::Material::SiO2, ps::Material::Mask).apply();
  return dom;
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

static ps::SmartPointer<cs::DenseCellSet<T, D>>
makeCells(ps::SmartPointer<ps::Domain<T, D>> dom) {
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., -(DEPTH + ROOM)}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deep,
        ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
  auto mm = ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::SiO2);
  for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto c = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
  c->setCellSetPosition(true);
  c->setCoverMaterial((int)ps::Material::GAS);
  c->fromLevelSets(lss, mm, MASKH + T(10));
  return c;
}

int main(int argc, char **argv) {
  if (argc > 1) TARGET = std::atof(argv[1]);
  if (argc > 2) SEED = (unsigned)std::atoi(argv[2]);
  if (argc > 3) DX = std::atof(argv[3]);
  if (const char *e = std::getenv("FCT_W")) W = std::atof(e);
  if (const char *e = std::getenv("FCT_MASK")) MASKH = std::atof(e);
  if (const char *e = std::getenv("FCT_DEPTH")) DEPTH = std::atof(e);
  if (const char *e = std::getenv("FCT_TAPER")) TAPER = std::atof(e);
  if (const char *e = std::getenv("FCT_ROOM")) ROOM = std::atof(e);
  const T scale = std::getenv("FCT_SCALE")
                      ? std::atof(std::getenv("FCT_SCALE")) : 1.0;
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");

  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/c2f6_sio2.mechanism.json");
  const auto gam = mech.sourceFluxes(ps::Material::SiO2);
  const auto kc = mech.rateConstantsFor(ps::Material::SiO2);
  std::vector<T> th(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gam, kc, th);
  const T ER = std::abs(mech.growthRate(gam, kc, th, ps::Material::SiO2));
  const T time = TARGET / ER;
  std::cout << std::fixed << std::setprecision(4)
            << "C2F6 / SiO2 trench: opening " << W << " nm, depth " << DEPTH
            << " nm, taper " << TAPER << " deg, mask " << MASKH
            << " nm, dx " << DX << " nm\n"
            << "blanket ER " << ER << " nm/s (" << ER * 60 << " nm/min),  t = "
            << time << " s for " << TARGET << " nm\n";
  for (size_t i = 0; i < th.size(); ++i)
    std::cout << "  theta_" << mech.coverageNames[i] << " = " << th[i] << "\n";
  std::cout << "\n";

  // ------------------------------------------------------------- level set
  if (!std::getenv("FCT_PMCONLY")) {
    auto dom = makeDomain(DX);
    // NO duplicated film layer. Carrying the polymer as a material puts every
    // surface point on Polymer, which switches off every reaction Table I
    // gates to SiO2, so nothing etches and the film seals the opening in 21 s.
    // It is carried as a THICKNESS instead, solved per point below.
    writeSurface(dom, "fct_ls_initial.vtp");
    auto model = ps::SmartPointer<ps::SurfaceChemistry<T, D>>::New(mech);
    {
      typename ps::impl::ChemicalSurfaceModel<T, D>::FilmRegulation reg;
      reg.solidIndex = 0;      // C, the fluorocarbon film
      reg.alpha = 0.6;         // Eq. (5)
      reg.gamma = 0.1;
      reg.monolayer = 10.0;    // 1e15 carbons per cm^2 in a 6 A monolayer
      reg.bulkSink = 9;        // F + C -> CF4, the bulk process of footnote (h)
      model->setFilmRegulation(reg);
    }
    ps::Process<T, D> proc(dom, model, time);
    proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
    ps::RayTracingParameters rt;
    rt.raysPerPoint = 500; rt.useRandomSeeds = false; rt.rngSeed = 1000;
    proc.setParameters(rt);
    ps::CoverageParameters cov; cov.tolerance = 1e-6; cov.maxIterations = 40;
    proc.setParameters(cov);
    std::cout << "level set:\n";
    proc.apply();
    writeSurface(dom, "fct_ls_final.vtp");
  }

  // ------------------------------------------------------- binary-cell PMC
  if (!std::getenv("FCT_LSONLY")) {
    auto cells = makeCells(makeDomain(DX));
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
    if (const char *e = std::getenv("FC_RANGE")) f.rangePerEv = std::atof(e);
    if (const char *e = std::getenv("FC_NRANGE")) f.neutralRange = std::atof(e);
    // see FCT_SCALE above: every probability up, the clock down
    f.stickOx *= scale; f.stickPoly *= scale; f.stickAct *= scale;
    f.kAct *= scale; f.pFPoly *= scale; f.pSputterP *= scale;
    f.pPolyWafer *= scale; f.pPassivate *= scale; f.pChemSputter *= scale;
    f.pDissociate *= scale; f.pFluor1 *= scale; f.pFluor2 *= scale;
    pmc.setSeed(SEED);
    pmc.setFluorocarbon(true, f);

    auto dump = [&](const std::string &name) {
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
      cells->writeVTU(name);
      std::cout << "    wrote " << name << "\n";
    };
    dump("fct_pmc_initial.vtu");
    const int steps = std::getenv("FCT_STEPS")
                          ? std::atoi(std::getenv("FCT_STEPS")) : 400;
    std::cout << "binary-cell PMC (probability scale " << scale << ", t = "
              << time / scale << " s):\n" << std::flush;
    for (int s = 0; s < steps; ++s)
      pmc.step(time / scale / steps);
    dump("fct_pmc_final.vtu");
    std::cout << "    polymer " << pmc.polymerMonolayers() << " ML,  Si events "
              << pmc.nOxEvents << " (ion " << pmc.nSiIon << " / F " << pmc.nSiF
              << " / P-wafer " << pmc.nSiPolyWafer << ")\n"
              << "    carbon in " << pmc.nPolyGrown << ", out "
              << std::llround(pmc.polyAtomsRemoved) << ", held "
              << std::llround(pmc.polymerCarbons()) << "\n";
  }
}
