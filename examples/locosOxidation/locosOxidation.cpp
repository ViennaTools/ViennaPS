// LOCOS (Local Oxidation of Silicon) process example using psOxidation.
//
// Geometry:
//   - Si substrate (plane at y = 0)
//   - Pad oxide (SiO2, thickness padOxideThickness) grown on Si before masking
//   - Si3N4 mask box (covering x < maskEdge, sitting on top of pad oxide)
//
// The Oxidation model auto-detects the Si3N4 material and activates LOCOS
// physics: mask bending + constrained-ambient advection, producing the
// characteristic bird's-beak oxide profile.
//
// Config keys (lengths in µm, time in hours, pressure in atm):
//   numThreads, gridDelta, xExtent, yMin, yMax,
//   padOxideThickness, maskThickness, maskEdge,
//   oxidationTime, timeStep, temperature, pressure, oxidant, orientation,
//   maxGridPoints, outputPrefix

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

#include <lsBooleanOperation.hpp>
#include <lsDomain.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsMakeGeometry.hpp>

#include <models/psOxidation.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

namespace ls = viennals;
namespace ps = viennaps;

using NumericType = double;
constexpr int D = 2;

// ── Geometry helpers ─────────────────────────────────────────────────────────

using LevelSet = ls::SmartPointer<ls::Domain<NumericType, D>>;
using BoundaryType = ls::Domain<NumericType, D>::BoundaryType;

LevelSet makePlane(const double *bounds, BoundaryType *bc,
                   NumericType gridDelta, NumericType y) {
  auto ls = ls::Domain<NumericType, D>::New(bounds, bc, gridDelta);
  const ls::VectorType<NumericType, D> origin{0., y};
  const ls::VectorType<NumericType, D> normal{0., 1.};
  ls::MakeGeometry<NumericType, D>(ls,
                                   ls::Plane<NumericType, D>::New(origin, normal))
      .apply();
  return ls;
}

LevelSet makeMask(const double *bounds, BoundaryType *bc, NumericType gridDelta,
                  NumericType xMin, NumericType xMax, NumericType yMin,
                  NumericType yMax) {
  auto mask = ls::Domain<NumericType, D>::New(bounds, bc, gridDelta);
  const ls::VectorType<NumericType, D> minCorner{xMin, yMin};
  const ls::VectorType<NumericType, D> maxCorner{xMax, yMax};
  ls::MakeGeometry<NumericType, D> geom(
      mask, ls::Box<NumericType, D>::New(minCorner, maxCorner));
  geom.setIgnoreBoundaryConditions(std::array<bool, D>{false, true});
  geom.apply();
  return mask;
}

// ── Config parser ────────────────────────────────────────────────────────────

struct Config {
  int numThreads = 4;
  NumericType gridDelta = 0.05;
  NumericType xExtent = 4.;
  NumericType yMin = -1.;
  NumericType yMax = 2.;
  NumericType padOxideThickness = 0.15;
  NumericType maskThickness = 0.2;
  NumericType maskEdge = 0.;
  NumericType oxidationTime = 0.35;
  NumericType timeStep = 0.1;
  NumericType temperature = 1000.;
  NumericType pressure = 1.;
  std::string oxidant = "wet";
  std::string orientation = "100";
  std::size_t maxGridPoints = 5000000;
  std::string outputPrefix = "ps_locos";
};

Config parseConfig(const std::string &filename) {
  Config cfg;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Warning: could not open " << filename
              << ", using defaults.\n";
    return cfg;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    const auto key = line.substr(0, eq);
    const auto val = line.substr(eq + 1);
    if (key == "numThreads")        cfg.numThreads = std::stoi(val);
    else if (key == "gridDelta")    cfg.gridDelta = std::stod(val);
    else if (key == "xExtent")      cfg.xExtent = std::stod(val);
    else if (key == "yMin")         cfg.yMin = std::stod(val);
    else if (key == "yMax")         cfg.yMax = std::stod(val);
    else if (key == "padOxideThickness") cfg.padOxideThickness = std::stod(val);
    else if (key == "maskThickness")cfg.maskThickness = std::stod(val);
    else if (key == "maskEdge")     cfg.maskEdge = std::stod(val);
    else if (key == "oxidationTime")cfg.oxidationTime = std::stod(val);
    else if (key == "timeStep")     cfg.timeStep = std::stod(val);
    else if (key == "temperature")  cfg.temperature = std::stod(val);
    else if (key == "pressure")     cfg.pressure = std::stod(val);
    else if (key == "oxidant")      cfg.oxidant = val;
    else if (key == "orientation")  cfg.orientation = val;
    else if (key == "maxGridPoints")cfg.maxGridPoints = std::stoull(val);
    else if (key == "outputPrefix") cfg.outputPrefix = val;
  }
  return cfg;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
  ps::Logger::setLogLevel(ps::LogLevel::ERROR);
  const auto cfg = parseConfig("config.txt");
  omp_set_num_threads(cfg.numThreads);

  const NumericType maskContactEpsilon = 1.e-6; // µm: mask bottom offset from oxide top

  double bounds[2 * D] = {-cfg.xExtent, cfg.xExtent, cfg.yMin, cfg.yMax};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  // Si substrate at y = 0.
  auto siLS = makePlane(bounds, bc, cfg.gridDelta, 0.);

  // Pad oxide: geometrically advance Si surface by padOxideThickness.
  auto oxLS = ls::Domain<NumericType, D>::New(siLS);
  auto sphere =
      ls::SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
          cfg.padOxideThickness);
  ls::GeometricAdvect<NumericType, D>(oxLS, sphere).apply();

  // SiN mask: box occupying x < maskEdge, sitting flat on the pad oxide.
  // The tiny contact epsilon places the mask bottom numerically below the oxide
  // top so Cartesian stencils unambiguously hit the mask boundary.
  auto maskLS =
      makeMask(bounds, bc, cfg.gridDelta, -cfg.xExtent, cfg.maskEdge,
               cfg.padOxideThickness - maskContactEpsilon,
               cfg.padOxideThickness + cfg.maskThickness);

  // ── ViennaPS domain ───────────────────────────────────────────────────────

  auto domain = ps::Domain<NumericType, D>::New();
  domain->insertNextLevelSetAsMaterial(siLS, ps::Material::Si, false);
  domain->insertNextLevelSetAsMaterial(oxLS, ps::Material::SiO2, false);
  domain->insertNextLevelSetAsMaterial(maskLS, ps::Material::Si3N4, false);

  // domain->saveSurfaceMesh(cfg.outputPrefix + "_stack_initial.vtp");
  domain->saveSurfaceMesh(cfg.outputPrefix + "_stack_step_0001.vtp");

  // ── Oxidation model ───────────────────────────────────────────────────────

  auto model = ps::SmartPointer<ps::Oxidation<NumericType, D>>::New();
  model->setTemperature(cfg.temperature);
  model->setOxidant(cfg.oxidant == "wet" ? ps::OxidantType::Wet
                                         : ps::OxidantType::Dry);
  model->setPressure(cfg.pressure);
  model->setOrientation(cfg.orientation == "111"
                            ? ps::SiliconOrientation::Si111
                            : ps::SiliconOrientation::Si100);
  model->setTimeStep(cfg.timeStep);
  model->setMaxGridPoints(cfg.maxGridPoints);

  // LOCOS: mask material is already Si3N4 (default); just set parameters.
  model->setMaskParameters(
      viennals::OxidationMaterials<NumericType>::siliconNitrideMask1000C());

  const NumericType est =
      model->estimatePlanarOxideThickness(cfg.padOxideThickness);
  std::cout << "Planar Deal-Grove estimate for " << cfg.oxidationTime
            << " hr oxidation at " << cfg.temperature
            << " C: " << est << " um total oxide thickness.\n";

  NumericType elapsed = 0.;
  unsigned step = 1;
  const NumericType timeEps = 1e-9 * cfg.oxidationTime;
  while (cfg.oxidationTime - elapsed > timeEps) {
    NumericType dt = cfg.timeStep;
    if (elapsed + dt > cfg.oxidationTime)
      dt = cfg.oxidationTime - elapsed;
    if (dt <= NumericType(0))
      break;

    model->setTime(dt);
    model->setTimeStep(dt);
    ps::Process<NumericType, D>(domain, model, NumericType(0)).apply();

    elapsed += dt;
    ++step;

    std::ostringstream filename;
    filename << cfg.outputPrefix << "_stack_step_" << std::setw(4)
             << std::setfill('0') << step << ".vtp";
    domain->saveSurfaceMesh(filename.str());
    std::cout << "Wrote " << filename.str() << " at t = " << elapsed
              << " hr.\n";
  }

  // ── Final output ──────────────────────────────────────────────────────────
  // The simulation level sets move independently (wrapLowerLevelSet = false),
  // which is required for LOCOS but means neither the surface mesh extractor
  // nor the volume mesh extractor sees the correct layered geometry directly.
  //
  // Surface mesh: union copies of the oxide and mask level sets to close the
  // sub-cell SDF-offset gap at their shared contact face before extraction.
  //
  // Volume mesh: the extractor assigns materials by layer order — it expects
  // each level set to enclose all lower ones.  Wrap copies (Si ⊂ oxide ⊂ mask)
  // so that material regions are correctly filled.  Do all of this on copies
  // so the live level sets are untouched for any further simulation steps.
  {
    const auto &ls_ = domain->getLevelSets();

    // Deep copies of each level set.
    LevelSet siCopy  = ls::Domain<NumericType, D>::New(ls_[0]);
    LevelSet oxCopy  = ls::Domain<NumericType, D>::New(ls_[1]);
    LevelSet mskCopy = ls::Domain<NumericType, D>::New(ls_[2]);

    // Surface mesh: close the oxide/mask gap with a simple union of ox and mask.
    {
      LevelSet oxSurf  = ls::Domain<NumericType, D>::New(ls_[1]);
      LevelSet mskSurf = ls::Domain<NumericType, D>::New(ls_[2]);
      ls::BooleanOperation<NumericType, D>(oxSurf, mskSurf,
                                           ls::BooleanOperationEnum::UNION).apply();
      auto surfDomain = ps::Domain<NumericType, D>::New();
      surfDomain->insertNextLevelSetAsMaterial(siCopy,  ps::Material::Si,    false);
      surfDomain->insertNextLevelSetAsMaterial(oxSurf,  ps::Material::SiO2,  false);
      surfDomain->insertNextLevelSetAsMaterial(mskSurf, ps::Material::Si3N4, false);
      surfDomain->saveSurfaceMesh(cfg.outputPrefix + "_stack_after.vtp");
    }

    // Volume mesh: wrap so that each LS encloses all lower ones.
    //   oxCopy  = UNION(Si, SiO2)   → SiO2 outer boundary wraps Si
    //   mskCopy = UNION(ox, Si3N4)  → Si3N4 outer boundary wraps Si + SiO2
    ls::BooleanOperation<NumericType, D>(oxCopy,  siCopy,
                                         ls::BooleanOperationEnum::UNION).apply();
    ls::BooleanOperation<NumericType, D>(mskCopy, oxCopy,
                                         ls::BooleanOperationEnum::UNION).apply();

    auto volDomain = ps::Domain<NumericType, D>::New();
    volDomain->insertNextLevelSetAsMaterial(siCopy,  ps::Material::Si,    false);
    volDomain->insertNextLevelSetAsMaterial(oxCopy,  ps::Material::SiO2,  false);
    volDomain->insertNextLevelSetAsMaterial(mskCopy, ps::Material::Si3N4, false);
    volDomain->saveVolumeMesh(cfg.outputPrefix + "_stack_after");
  }

  std::cout << "Wrote " << cfg.outputPrefix << "_stack_initial.vtp, "
            << step << " time-step files, "
            << cfg.outputPrefix << "_stack_after.vtp, and "
            << cfg.outputPrefix << "_stack_after_volume.vtu\n";

  return 0;
}
