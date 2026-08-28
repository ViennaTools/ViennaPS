// The same chemistry, the same feature, two surface representations.
//
// A level set carries the surface as a sub-grid quantity and advects it. A
// voxel grid carries filling fractions and moves material between cells. The
// chemistry is not a variable here: both arms construct the SAME
// ChemicalMechanism from the same file, solve the same coverages and ask it
// for the same velocity. What differs is how a ray finds the surface and what
// normal it meets there.
//
// The blanket is checked first, because it has an analytic answer and because
// a difference in a trench means nothing until the two agree where there is no
// geometry to disagree about. Only then is the trench worth reading.

#include <models/psChemicalMechanismIO.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeTrench.hpp>
#include <process/psProcess.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psVoxelChemistry.hpp>

#include <lsMakeGeometry.hpp>
#include <lsToSurfaceMesh.hpp>

#include <cmath>
#include <iomanip>
#include <chrono>
#include <iostream>

namespace ls = viennals;
namespace cs = viennacs;
namespace ps = viennaps;

using T = double;
constexpr int D = 2;

#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

// BOTH arms must measure the same quantity, or the comparison measures the
// probes. That quantity is: at a given lateral band, the highest point at
// which material exists. On the open field that is the top of the film; at the
// trench centre it is the floor, because the sidewalls lie outside the band.
// It is well defined for a deposit and for an etch, and it does not care how
// the surface is stored.
//
// The bands are placed away from the sidewalls so neither probe ever catches a
// wall node or a wall cell.
struct Bands {
  T fieldLo, fieldHi;   // open field, outside the trench
  T floorLo, floorHi;   // trench centre
};

inline Bands makeBands(T width, T domainWidth) {
  const T half = width / 2;
  return Bands{-T(0.45) * domainWidth, -T(0.30) * domainWidth,
               -T(0.35) * half, T(0.35) * half};
}

struct Profile {
  T fieldChange = 0;  ///< how far the flat top moved
  T bottomChange = 0; ///< how far the trench floor moved
  T stepCoverage = 0; ///< bottom over field
};

/// The level-set arm: ViennaPS as it stands.
Profile levelSetArm(ps::ChemicalMechanism<T> mech, T time, T gridDelta,
                    T width, T depth, size_t rays, T maskHeight = 0,
                    bool maskOnly = false,
                    ps::FluxEngineType engine = ps::FluxEngineType::CPU_TRIANGLE) {
  // maskOnly: a FLAT substrate under the mask -- the slot in the mask is the
  // only opening, and the etch digs the trench itself, as a real masked etch
  // does. `depth` then only sizes probes and margins, not the geometry.
  const T trenchD = maskOnly ? T(0) : depth;
  auto domain = ps::SmartPointer<ps::Domain<T, D>>::New(gridDelta, T(4) * width,
                                                        T(4) * width);
  // width, depth, taper, maskHeight, maskTaper, halfTrench, material
  ps::MakeTrench<T, D>(domain, width, trenchD, T(0), maskHeight, T(0), false,
                       ps::Material::Si, ps::Material::Mask)
      .apply();

  const auto bands = makeBands(width, 4 * width);
  auto bounds = [&]() {
    auto mesh = ps::SmartPointer<ls::Mesh<T>>::New();
    ls::ToSurfaceMesh<T, D>(domain->getLevelSets().back(), mesh).apply();
    T field = std::numeric_limits<T>::lowest(),
      floor = std::numeric_limits<T>::lowest();
    for (const auto &n : mesh->getNodes()) {
      if (n[0] >= bands.fieldLo && n[0] <= bands.fieldHi)
        field = std::max(field, n[1]);
      if (n[0] >= bands.floorLo && n[0] <= bands.floorHi)
        floor = std::max(floor, n[1]);
    }
    return std::pair<T, T>{field, floor};
  };

  const auto before = bounds();
  domain->saveSurfaceMesh(mech.name + (maskHeight > 0 ? "_masked" : "") +
                          "_ls_initial.vtp");
  auto model = ps::SmartPointer<ps::SurfaceChemistry<T, D>>::New(mech);
  ps::Process<T, D> process(domain, model, time);
  process.setFluxEngineType(engine);
  ps::RayTracingParameters rt;
  rt.raysPerPoint = static_cast<unsigned>(rays);
  process.setParameters(rt);
  ps::CoverageParameters cov;
  cov.tolerance = 1e-6;
  cov.maxIterations = 40;
  process.setParameters(cov);
  { const auto t0=std::chrono::steady_clock::now();
    process.apply();
    std::cout<<"    [time] level-set arm process: "<<std::fixed
             <<std::setprecision(1)<<std::chrono::duration<double>(
                 std::chrono::steady_clock::now()-t0).count()<<" s"<<std::endl; }
  const auto after = bounds();
  domain->saveSurfaceMesh(mech.name + (maskHeight > 0 ? "_masked" : "") +
                          "_ls_final.vtp");

  Profile p;
  p.fieldChange = after.first - before.first;
  p.bottomChange = after.second - before.second;
  p.stepCoverage =
      std::abs(p.fieldChange) > 1e-12 ? p.bottomChange / p.fieldChange : 0;
  return p;
}

/// The voxel arm: the same mechanism on a cell set built from the same
/// geometry, then evolved without the level sets being consulted again.
Profile voxelArm(ps::ChemicalMechanism<T> mech, T time, T gridDelta, T width,
                 T depth, size_t rays, int steps,
                 cs::NormalEstimator estimator, T maskHeight = 0,
                 bool maskOnly = false) {
  const T trenchD = maskOnly ? T(0) : depth;
  auto domain = ps::SmartPointer<ps::Domain<T, D>>::New(gridDelta, T(4) * width,
                                                        T(4) * width);
  // width, depth, taper, maskHeight, maskTaper, halfTrench, material
  ps::MakeTrench<T, D>(domain, width, trenchD, T(0), maskHeight, T(0), false,
                       ps::Material::Si, ps::Material::Mask)
      .apply();
  // The cell set needs room on BOTH sides of the surface: a gas cover above,
  // or a deposit has nowhere to grow, and a solid margin below the deepest
  // surface point, or an etch runs out of lattice. generateCellSet bounds the
  // set one row under the deepest surface, which starved the SF6/O2 floor: the
  // bottom row's interface area, taken against the clamped lattice boundary,
  // decays with its own fill, and the floor froze while the field etched on.
  // An auxiliary plane below everything the process will reach provides the
  // margin; the material map is required, since without one the cover material
  // is never applied and the gas region reads as solid.
  auto topLS = domain->getLevelSets().back();
  auto deepPlane = ls::SmartPointer<ls::Domain<T, D>>::New(topLS->getGrid());
  {
    T o[D] = {0., -(depth + T(8))}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deepPlane,
                           ls::SmartPointer<ls::Plane<T, D>>::New(o, n))
        .apply();
  }
  // Every level set the domain holds, with ITS material -- a mask is a
  // second material, and hardcoding Si here is how a mask silently becomes
  // substrate. The deep margin plane leads, as more substrate.
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deepPlane};
  auto matMap = ls::SmartPointer<ls::MaterialMap>::New();
  matMap->insertNextMaterial(static_cast<int>(ps::Material::Si));
  for (size_t l = 0; l < domain->getLevelSets().size(); ++l) {
    lss.push_back(domain->getLevelSets()[l]);
    matMap->insertNextMaterial(static_cast<int>(
        domain->getMaterialMap()->getMaterialAtIdx(l)));
  }
  auto cellSet = viennacore::SmartPointer<cs::DenseCellSet<T, D>>::New();
  cellSet->setCellSetPosition(true);
  cellSet->setCoverMaterial(static_cast<int>(ps::Material::GAS));
  // The gas cover must clear the TALLEST surface, which with a mask is the
  // mask top -- a cover plane below it leaves the slot's gas column with no
  // cells at all, a hole in the lattice at the trench mouth, and rays cross
  // it uninterrupted until they strike the mask from inside.
  cellSet->fromLevelSets(lss, matMap, maskHeight + T(6));

  cs::LatticeMap<T, D> lattice(*cellSet);
  const auto &materialIds = *cellSet->getScalarData("Material");
  std::vector<int> material(cellSet->getNumberOfCells());
  for (size_t c = 0; c < material.size(); ++c)
    material[c] = static_cast<int>(materialIds[c]);

  // Start from the level set's own sub-grid surface, so step zero is the same
  // geometry in both arms rather than a staircase in one of them.
  std::vector<T> fill(cellSet->getNumberOfCells(), T(0));
  for (size_t c = 0; c < fill.size(); ++c)
    fill[c] = material[c] == static_cast<int>(ps::Material::GAS) ? T(0) : T(1);

  if(maskHeight>T(0)){ // a masked run whose initial state has no mask is wrong
    size_t nm=0;
    for(size_t c=0;c<fill.size();++c)
      if(material[c]==(int)ps::Material::Mask && fill[c]>=T(0.5)) ++nm;
    if(nm==0){ std::cerr<<"FATAL: masked geometry but no Mask cells at t=0\n"; std::abort(); }
  }
  ps::VoxelChemistry<T, D> voxel(mech, lattice, fill, material);
  // Without this the estimator argument is accepted and ignored, and the two
  // voxel rows of every table are the same run twice.
  voxel.setNormalEstimator(estimator);
  voxel.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
  voxel.setRaysPerStep(rays * 200);
  auto coverages = voxel.makeCoverages();

  const auto bands = makeBands(width, 4 * width);
  auto surface = [&](bool field) {
    // The highest point holding material, averaged over the band -- the same
    // quantity the level-set probe takes from its surface mesh. The topmost
    // partial cell is resolved within itself by its own fraction, so this is
    // sub-grid rather than quantised to a cell face.
    const auto &dims = lattice.dims();
    const T lo = field ? bands.fieldLo : bands.floorLo;
    const T hi = field ? bands.fieldHi : bands.floorHi;
    T sum = 0;
    int counted = 0;
    for (int i = 0; i < dims[0]; ++i) {
      const T x = lattice.minCorner()[0] + gridDelta * (T(i) + T(0.5));
      if (x < lo || x > hi)
        continue;
      for (int j = dims[1] - 1; j >= 0; --j) {
        const int id = lattice.cellId({i, j});
        if (id < 0 || fill[id] <= T(1e-6))
          continue;
        const T cellTop = lattice.minCorner()[1] + gridDelta * T(j + 1);
        sum += cellTop - (T(1) - fill[id]) * gridDelta;
        ++counted;
        break;
      }
    }
    return counted ? sum / counted : T(0);
  };

  // The evolved state, visualisable: the run's own fill fractions written
  // back into the cell set, whose grid the VTK writer already understands.
  // In ParaView, threshold FillingFraction >= 0.5 for the solid, or volume
  // render the fraction itself to see the sub-cell interface.
  auto writeCells = [&](const std::string &fileName) {
    auto &ff = *cellSet->getFillingFractions();
    auto &mm = *cellSet->getScalarData("Material");
    const auto &labels = voxel.materials(); // the EVOLVED labels
    for (size_t c = 0; c < fill.size(); ++c) {
      ff[c] = fill[c];
      mm[c] = static_cast<T>(labels[c]);
    }
    cellSet->writeVTU(fileName);
  };
  const std::string tag =
      std::string(maskHeight > 0 ? "masked_" : "") +
      (estimator == cs::NormalEstimator::Face ? "face" : "youngs");
  writeCells(mech.name + (maskHeight > 0 ? "_masked" : "") +
             "_voxel_initial.vtu");
  const T field0 = surface(true), bottom0 = surface(false);
  double tTr=0,tCh=0,tAd=0,tRe=0;
  for (int s = 0; s < steps; ++s) {
    const auto r = voxel.step(time / steps, coverages, 1 + s);
    tTr+=r.secondsTransport; tCh+=r.secondsChemistry;
    tAd+=r.secondsAdvance; tRe+=r.secondsRelabel;
  }
  std::cout<<"    [time] voxel arm, "<<steps<<" steps: transport "
           <<std::fixed<<std::setprecision(1)<<tTr<<" s, chemistry "<<tCh
           <<" s, advance "<<tAd<<" s, relabel "<<tRe<<" s  (total "
           <<tTr+tCh+tAd+tRe<<" s)"<<std::endl;
  writeCells(mech.name + "_voxel_" + tag + "_final.vtu");

  Profile p;
  p.fieldChange = surface(true) - field0;
  p.bottomChange = surface(false) - bottom0;
  p.stepCoverage =
      std::abs(p.fieldChange) > 1e-12 ? p.bottomChange / p.fieldChange : 0;
  return p;
}

// NOTE ON IONS. The voxel arm does not trace an ion: an ion yield channel
// takes the analytic normal-incidence yield, which is exact on a blanket and
// unshadowed in a feature. Comparing an ion mechanism in a trench would
// therefore measure that gap and nothing else, so the mechanisms here are
// ion-free. An ion particle for the voxel arm -- carrying energy through
// reflections and evaluating the yield against the local angle of incidence --
// is the next piece of work, and it is what the normal estimators exist for.
int main(int argc, char **argv) {
  std::vector<std::string> files;
  if (argc > 1)
    for (int i = 1; i < argc; ++i)
      files.emplace_back(argv[i]);
  else
    for (const char *m : {"silane", "sige_stack", "diamond"})
      files.emplace_back(std::string(VIENNAPS_MECHANISM_DIR) + "/" + m +
                         ".mechanism.json");

  ps::Logger::setLogLevel(ps::LogLevel::WARNING);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");

  for (const auto &file : files) {
  auto mech = ps::readChemicalMechanism<T>(file);

  const auto material = ps::Material::Si;
  const auto gamma = mech.sourceFluxes(material);
  const auto k = mech.rateConstantsFor(material);
  std::vector<T> theta(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gamma, k, theta);
  const T analytic = mech.growthRate(gamma, k, theta, material);

  std::cout << "Level set against voxels: " << mech.name << "\n";
  std::cout << "  analytic blanket rate " << std::scientific
            << std::setprecision(4) << analytic << " nm/s\n\n";

  const T gridDelta = 1.0, width = 40., depth = 60.;
  // Deposit a fraction of the trench half-width, or the feature closes and
  // "step coverage" stops meaning anything: near-conformal growth of 20 nm
  // puts 20 nm on EACH wall and pinches a 40 nm trench shut, after which the
  // floor probe reports material rising past the old field level. 8 nm leaves
  // the trench open and the ratio measurable.
  const T target = T(0.2) * width / 2;
  const T time = target / std::abs(analytic);

  const auto lsArm = levelSetArm(mech, time, gridDelta, width, depth, 500);
  const auto vxFace =
      voxelArm(mech, time, gridDelta, width, depth, 500, 20,
               cs::NormalEstimator::Face);
  const auto vxYoungs =
      voxelArm(mech, time, gridDelta, width, depth, 500, 20,
               cs::NormalEstimator::FillGradientYoungs);

  std::cout << "  arm                     field [nm]   bottom [nm]   step "
               "coverage\n";
  auto row = [](const std::string &name, const Profile &p) {
    std::cout << "  " << std::left << std::setw(22) << name << std::right
              << std::fixed << std::setprecision(3) << std::setw(11)
              << p.fieldChange << std::setw(14) << p.bottomChange
              << std::setw(15) << p.stepCoverage << "\n";
  };
  row("level set", lsArm);
  row("voxel, face normal", vxFace);
  row("voxel, Youngs normal", vxYoungs);

  std::cout << "\n  field against analytic:  level set "
            << std::setprecision(1)
            << 100 * (lsArm.fieldChange / (analytic * time) - 1) << "%"
            << ",  voxel " << 100 * (vxYoungs.fieldChange / (analytic * time) - 1)
            << "%\n"
            << "  step coverage, voxel against level set:  "
            << std::setprecision(1)
            << 100 * (vxYoungs.stepCoverage / lsArm.stepCoverage - 1) << "%\n\n";

  // An etch gets the masked variant too: the mask is where the per-material
  // physics lives -- rho(Mask) = 500 and its own yields make it erode a
  // hundred times slower -- and a bare trench never exercises any of it.
  if (analytic < 0) {
    const T maskH = 30.;
    const T timeM = 5 * time; // dig ~20 nm: topography, not a one-cell dent
    std::cout << "  with a " << maskH
              << " nm mask on a FLAT substrate, 5x the time:\n";
    std::cout << "  arm                  mask top [nm]   floor [nm]\n";
    auto mrow = [](const std::string &n, const Profile &p) {
      std::cout << "  " << std::left << std::setw(20) << n << std::right
                << std::fixed << std::setprecision(3) << std::setw(12)
                << p.fieldChange << std::setw(13) << p.bottomChange << "\n";
    };
    const auto lsM = levelSetArm(mech, timeM, gridDelta, width, depth, 500,
                                 maskH, true);
    const auto vxMf = voxelArm(mech, timeM, gridDelta, width, depth, 500, 100,
                               cs::NormalEstimator::Face, maskH, true);
    const auto vxM = voxelArm(mech, timeM, gridDelta, width, depth, 500, 100,
                              cs::NormalEstimator::FillGradientYoungs, maskH,
                              true);
    mrow("level set", lsM);
    mrow("voxel, face", vxMf);
    mrow("voxel, Youngs", vxM);
    std::cout << "  floor, voxel against level set: " << std::setprecision(1)
              << 100 * (vxM.bottomChange / lsM.bottomChange - 1) << "%\n\n";
  }
  }
  return 0;
}
