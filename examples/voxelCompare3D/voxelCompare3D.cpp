// The three surface representations on a 3D masked HOLE.
//
// 2D cannot pose this: a hole shadows in both lateral directions at once, so
// the neutral supply to its floor falls off far faster than in a trench of the
// same width. It is also the case where the binary-cell PMC is cheapest to get
// wrong -- the staircase now has side faces on four sides of every column.
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>
#include <models/psVoxelPMC.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeHole.hpp>
#include <process/psProcess.hpp>

#include <csDenseCellSet.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace ls = viennals; namespace cs = viennacs; namespace ps = viennaps;
using T = double; constexpr int D = 3;
#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

// Geometry, overridable so the feature can be scaled against the normal-fit
// stencil: a fit radius wider than the hole radius smooths the hole away.
// H3_R / H3_EXT / H3_MASK / H3_DEEP, in nm.
static T DX = 1.0, EXT = 40.0, RADIUS = 8.0, MASKH = 20.0, DEEP = -18.0;
static T TARGET = 5.0;

static ps::SmartPointer<ps::Domain<T, D>> makeDomain() {
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(DX, EXT, EXT);
  ps::MakeHole<T, D>(dom, RADIUS, T(0), T(0), MASKH, T(0),
                     ps::HoleShape::FULL, ps::Material::Si,
                     ps::Material::Mask).apply();
  return dom;
}

static ps::SmartPointer<cs::DenseCellSet<T, D>>
makeCells(ps::SmartPointer<ps::Domain<T, D>> dom) {
  auto top = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(top->getGrid());
  { T o[D] = {0., 0., DEEP}, n[D] = {0., 0., 1.};
    ls::MakeGeometry<T, D>(deep,
        ls::SmartPointer<ls::Plane<T, D>>::New(o, n)).apply(); }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
  auto mm = ls::SmartPointer<ls::MaterialMap>::New();
  mm->insertNextMaterial((int)ps::Material::Si);
  for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
    lss.push_back(dom->getLevelSets()[l]);
    mm->insertNextMaterial((int)dom->getMaterialMap()->getMaterialAtIdx(l));
  }
  auto c = ps::SmartPointer<cs::DenseCellSet<T, D>>::New();
  c->setCellSetPosition(true);
  c->setCoverMaterial((int)ps::Material::GAS);
  c->fromLevelSets(lss, mm, MASKH + T(4));
  return c;
}

// mean height of the solid column tops within half the hole radius
static T floorDepth(cs::LatticeMap<T, D> &lat, const std::vector<T> &fill,
                    const std::vector<int> &material, T thresh) {
  const auto &dims = lat.dims();
  T sum = 0; int n = 0;
  for (int i = 0; i < dims[0]; ++i)
    for (int j = 0; j < dims[1]; ++j) {
      const T x = lat.minCorner()[0] + DX * (i + T(0.5));
      const T y = lat.minCorner()[1] + DX * (j + T(0.5));
      if (x * x + y * y > (RADIUS / 2) * (RADIUS / 2)) continue;
      for (int k = dims[2] - 1; k >= 0; --k) {
        const int id = lat.cellId({i, j, k});
        if (id < 0 || material[id] == (int)ps::Material::Mask ||
            fill[id] <= thresh) continue;
        sum += lat.minCorner()[2] + DX * (k + 1) - (T(1) - fill[id]) * DX;
        ++n; break;
      }
    }
  return n ? -sum / n : T(0);
}

int main(int argc, char **argv) {
  if (argc > 1) TARGET = std::atof(argv[1]);
  if (const char *e = std::getenv("H3_R")) RADIUS = std::atof(e);
  if (const char *e = std::getenv("H3_EXT")) EXT = std::atof(e);
  if (const char *e = std::getenv("H3_MASK")) MASKH = std::atof(e);
  if (const char *e = std::getenv("H3_DEEP")) DEEP = std::atof(e);
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");
  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");
  const auto gam = mech.sourceFluxes(ps::Material::Si);
  const auto kc = mech.rateConstantsFor(ps::Material::Si);
  std::vector<T> th(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gam, kc, th);
  const T ER = std::abs(mech.growthRate(gam, kc, th, ps::Material::Si));
  const T time = TARGET / ER;
  std::cout << std::fixed << std::setprecision(3)
            << "3D hole: r=" << RADIUS << " mask=" << MASKH << " dx=" << DX
            << " extent=" << EXT << ",  t=" << time << " s (" << TARGET
            << " nm blanket-equivalent)\n\n";
  auto tick = [](const char *what, auto fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    std::cout << "    [" << std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count() << " s]  " << what << "\n";
  };

  { auto dom = makeDomain();
    auto model = ps::SmartPointer<ps::SurfaceChemistry<T, D>>::New(mech);
    ps::Process<T, D> proc(dom, model, time);
    proc.setFluxEngineType(ps::FluxEngineType::CPU_TRIANGLE);
    ps::RayTracingParameters rt;
    rt.raysPerPoint = 200; rt.useRandomSeeds = false; rt.rngSeed = 1000;
    proc.setParameters(rt);
    ps::CoverageParameters cov; cov.tolerance = 1e-6; cov.maxIterations = 40;
    proc.setParameters(cov);
    std::cout << "level set\n";
    tick("apply", [&]{ proc.apply(); });
    dom->saveSurfaceMesh("h3_ls_final.vtp");
    auto cells = makeCells(dom);
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> f(cells->getNumberOfCells()), m(cells->getNumberOfCells());
    std::vector<int> mi(cells->getNumberOfCells());
    for (size_t c = 0; c < f.size(); ++c) {
      mi[c] = (int)mid[c];
      f[c] = mi[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    std::cout << "    floor " << floorDepth(lat, f, mi, 1e-6) << " nm\n\n";
  }

  auto voxelRun = [&](bool pmc) {
    auto cells = makeCells(makeDomain());
    cs::LatticeMap<T, D> lat(*cells);
    const auto &mid = *cells->getScalarData("Material");
    std::vector<T> fill(cells->getNumberOfCells(), T(0));
    std::vector<int> material(cells->getNumberOfCells());
    for (size_t c = 0; c < fill.size(); ++c) {
      material[c] = (int)mid[c];
      fill[c] = material[c] == (int)ps::Material::GAS ? T(0) : T(1);
    }
    // binary arms cut at 0.5; the filling-fraction arm keeps its graded band
    // create the field BEFORE any reference into the cell data is
    // taken: adding one reallocates and would dangle them
    cells->addScalarData("State", 0.);
    auto dump = [&](const std::string &name,
                    const std::vector<std::uint8_t> *st, T solidAbove) {
      auto &ff = *cells->getFillingFractions();
      auto &mmv = *cells->getScalarData("Material");
      auto &state = *cells->getScalarData("State");
      for (size_t c = 0; c < fill.size(); ++c) {
        ff[c] = fill[c];
        // Material stays the material; the PMC's chemical state is its own
        // field. The writer drops the gas on its own.
        // State is the field to colour by: material AND chemistry in one.
        //   0 bare Si | 1 fluorinated | 2 oxidised | 3 mask
        // Material stays the plain ViennaPS id, for analysis.
        const bool solid = fill[c] >= solidAbove;
        const bool mask = material[c] == (int)ps::Material::Mask;
        mmv[c] = solid ? T(material[c]) : T((int)ps::Material::GAS);
        state[c] = !solid ? T(0) : (mask ? T(3) : (st ? T((*st)[c]) : T(0)));
      }
      cells->writeVTU(name);
    };
    if (!pmc) {
      ps::VoxelChemistry<T, D> vox(mech, lat, fill, material);
      vox.setRaysPerCell(200);
      vox.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
      if (const char *e = std::getenv("VOX_SPREAD"))
        vox.setSurplusSpreading(std::atoi(e));
      auto cov = vox.makeCoverages();
      vox.initialiseCoverages(cov, 1000u, 100, T(1e-6));
      const int steps = std::max(50, (int)std::lround(200 * TARGET / 5.0));
      std::cout << "voxel, filling fraction\n";
      tick((std::to_string(steps) + " steps").c_str(), [&]{
        for (int s = 0; s < steps; ++s)
          vox.step(time / steps, cov, 100003u + s); });
      dump("h3_voxel_final.vtu", nullptr, T(1e-6));
      std::cout << "    floor " << floorDepth(lat, fill, material, 1e-6) << " nm\n\n";
    } else {
      ps::VoxelPMC<T, D> p(lat, fill, material);   // library defaults
      if (const char *e = std::getenv("PMC_FITR")) p.setFitRadius(std::atoi(e));
      if (const char *e = std::getenv("PMC_REFLR")) p.setReflectRadius(std::atoi(e));
      if (std::getenv("PMC_NOCAP")) p.setCurvatureCap(false);
      p.setSeed(7);
      const int steps = std::max(200, (int)std::lround(600 * TARGET / 5.0));
      std::cout << "binary-cell PMC (defaults: re-emission, accumulator, M=4)\n";
      tick((std::to_string(steps) + " steps").c_str(), [&]{
        for (int s = 0; s < steps; ++s) p.step(time / steps); });
      dump("h3_pmc_final.vtu", &p.states(), T(0.5));
      const auto c = p.coverages();
      std::cout << "    floor " << floorDepth(lat, fill, material, 0.5)
                << " nm   theta_F " << c[0] << "  theta_O " << c[1]
                << "   removed " << p.removedCells() << "\n"
                << "    bounces " << p.nBounce << "  could-not-place "
                << p.nBounceFail << "  escaped/inside " << p.nEscape
                << "  hit cap " << p.nBounceCap
                << "   adsorbed F " << p.nAdsF << "\n";
    }
  };
  voxelRun(false);
  if (!std::getenv("H3_NOPMC"))
    voxelRun(true);
}
