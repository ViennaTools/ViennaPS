// P implantation in Si — analytic (manual) mode only.
//
// All process parameters are read from config.txt.  The Pearson IV moments,
// damage parameters, and anneal physics are taken verbatim from the ViennaPS
// modeldb tables (see comments in config.txt), so no modeldb lookup happens
// at runtime.
//
// Usage: pImplantManual [config.txt]
//
// Output VTU files (open in ParaView):
//   initial.vtu        — geometry + material IDs
//   post_implant.vtu   — dopant total concentration + damage field
//   post_anneal.vtu    — dopant total/active concentration + I/V fields
//
// Output CSV depth profiles:
//   profile_post_implant.csv — dopant + damage vs depth
//   profile_post_anneal.csv  — dopant total/active + I/V vs depth

#include "../ionImplantation/exampleConfig.hpp"
#include <psDomain.hpp>
#include <process/psProcess.hpp>

#include <lsBooleanOperation.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>

#include <vcUtil.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace viennaps;

// Writes a CSV with the vertical depth profile of the given fields. ViennaPS
// uses y < 0 inside the substrate; this writer reports positive depth from the
// wafer surface at y = 0.
// For each depth slice, outputs the sum and peak (max) across all x-cells.
// The "sum" column integrates the concentration laterally at each depth —
// useful for checking conserved dose; the "max" column gives the peak
// concentration at that depth (would match a 1-D full-dose simulation at x=0).
// Note: the domain uses a reflective boundary at x=0, so the physical half-width
// is xExtent/2; the sum therefore represents the half-domain integral.
template <typename T, int D>
static void writeDepthProfile(
    const SmartPointer<viennaps::Domain<T, D>> &domain,
    const std::vector<std::string> &labels,
    const std::string &filename) {

  auto cs = domain->getCellSet();
  const std::size_t nCells = cs->getNumberOfCells();
  const T delta = cs->getGridDelta();

  // Collect field pointers (nullptr when a field has not been written yet).
  std::vector<const std::vector<T> *> fields;
  fields.reserve(labels.size());
  for (const auto &lbl : labels)
    fields.push_back(cs->getScalarData(lbl));

  // Bin cells by positive substrate depth, rounding to the nearest grid point
  // so floating-point coordinates land on the same key.
  struct Bin {
    std::size_t count = 0;
    std::vector<T> sum;
    std::vector<T> max;
    explicit Bin(std::size_t n) : sum(n, T(0)), max(n, T(0)) {}
  };
  std::map<T, Bin> bins;

  for (std::size_t idx = 0; idx < nCells; ++idx) {
    const auto center = cs->getCellCenter(idx);
    const T depth = -center[D - 1];
    if (depth < T(0))
      continue;
    const T depthKey = std::round(depth / delta) * delta;

    auto [it, inserted] = bins.emplace(depthKey, Bin(labels.size()));
    auto &bin = it->second;
    bin.count++;
    for (std::size_t f = 0; f < labels.size(); ++f) {
      if (!fields[f]) continue;
      const T v = (*fields[f])[idx];
      bin.sum[f] += v;
      bin.max[f] = std::max(bin.max[f], v);
    }
  }

  std::ofstream out(filename);
  out << "# Vertical depth profile\n";
  out << "# depth_nm: positive distance into substrate (surface y=0 minus cell centre y)\n";
  out << "# _sum: sum of field over all x-cells at this depth\n";
  out << "# _max: peak field value across x at this depth\n";
  out << "depth_nm";
  for (const auto &lbl : labels)
    out << "," << lbl << "_sum," << lbl << "_max";
  out << "\n";

  for (const auto &[depth, bin] : bins) {
    out << depth;
    for (std::size_t f = 0; f < labels.size(); ++f)
      out << "," << bin.sum[f] << "," << bin.max[f];
    out << "\n";
  }

  std::cout << "  wrote: " << filename << " (" << bins.size()
            << " depth slices)\n";
}

int main(int argc, char *argv[]) {
  using T = double;
  constexpr int D = 2;

  const std::string cfgPath = argc > 1 ? argv[1] : "config.txt";

  util::Parameters params;
  params.readConfigFile(cfgPath);
  const auto rawParams = ionimpl::readRawParameters(cfgPath);
  if (params.m.empty()) {
    std::cerr << "Config not found: " << cfgPath << "\n";
    std::cerr << "Usage: " << argv[0] << " [config.txt]\n";
    return 1;
  }

  std::cout << "--- ViennaPS Manual Implant & Anneal (config: " << cfgPath
            << ") ---\n";

  // ── Geometry parameters ───────────────────────────────────────────────────
  const T gridDelta      = params.get("gridDelta");
  const T xExtent        = params.get("xExtent");
  const T topSpace       = params.get("topSpace");
  const T substrateDepth = params.get("substrateDepth");
  const T openingWidth   = params.get("openingWidth");
  const T maskHeight     = params.get("maskHeight");
  const T oxideThickness = params.get("oxideThickness");
  const T screenThickness =
      params.m.count("screenThickness") ? params.get("screenThickness")
                                        : oxideThickness;

  T bounds[2 * D] = {-0.5 * xExtent, 0.5 * xExtent, -substrateDepth,
                     topSpace + oxideThickness + maskHeight};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  auto domain = Domain<T, D>::New(bounds, bc, gridDelta);

  auto makels = [&]() {
    return SmartPointer<viennals::Domain<T, D>>::New(bounds, bc, gridDelta);
  };

  // Si substrate bottom
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = -substrateDepth;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls, viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  // Si substrate top (surface at y = 0)
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls, viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  // Screen oxide (y = 0 to y = oxideThickness)
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = oxideThickness;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls, viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::SiO2);
  }
  // Hard mask with opening
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = oxideThickness + maskHeight;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls, viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);

    auto window = makels();
    T wMin[D] = {-0.5 * openingWidth, oxideThickness - gridDelta};
    T wMax[D] = {0.5 * openingWidth, oxideThickness + maskHeight + gridDelta};
    viennals::MakeGeometry<T, D>(window, viennals::Box<T, D>::New(wMin, wMax))
        .apply();
    domain->applyBooleanOperation(
        window, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  domain->generateCellSet(topSpace, Material::Air, /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  domain->getCellSet()->writeVTU("initial.vtu");

  // ── Process models ────────────────────────────────────────────────────────
  auto implant = SmartPointer<IonImplantation<T, D>>::New();
  auto anneal  = SmartPointer<Anneal<T, D>>::New();

  const auto annealSchedule = ionimpl::readAnnealSchedule<T>(rawParams);
  const T peakT = viennaps::peakAnnealTemperature(annealSchedule);

  const auto implantConfig =
      ionimpl::makeAnalyticImplantSetup<T, D>(params, screenThickness);
  std::cout << "Implanting " << implantConfig.description << " ...\n";
  viennaps::applyImplantSetup(*implant, implantConfig);

  std::cout << "Annealing: peak T = " << (peakT - T(273.15)) << " C ...\n";
  const auto annealConfig = ionimpl::makeAnnealSetup<T>(
      params, annealSchedule, implantConfig, peakT,
      {viennaps::Material::Si},
      {viennaps::Material::Mask, viennaps::Material::SiO2},
      /*defaultUseModelDb=*/false);
  std::cout << "Anneal parameter source: " << annealConfig.model.source << "\n";
  viennaps::applyAnnealSetup(*anneal, annealConfig);

  // ── Run and write profiles ────────────────────────────────────────────────
  Process<T, D>(domain, implant, T(0)).apply();
  domain->getCellSet()->writeVTU("post_implant.vtu");
  writeDepthProfile<T, D>(domain,
      {implantConfig.labels.total, implantConfig.labels.damage},
      "profile_post_implant.csv");

  Process<T, D>(domain, anneal, T(0)).apply();
  domain->getCellSet()->writeVTU("post_anneal.vtu");
  writeDepthProfile<T, D>(domain,
      {implantConfig.labels.total, implantConfig.labels.active,
       implantConfig.labels.damage, implantConfig.labels.interstitial,
       implantConfig.labels.vacancy},
      "profile_post_anneal.csv");

  // ── Stats ─────────────────────────────────────────────────────────────────
  std::cout << "\n--- POST-ANNEAL STATS ---\n";
  auto printFieldStats = [&](const std::string &label) {
    if (label.empty()) return;
    auto field = domain->getCellSet()->getScalarData(label);
    if (!field) { std::cout << label << " <missing>\n"; return; }
    T maxVal = 0.;
    long double sumVal = 0.;
    for (const auto &v : *field) {
      maxVal = std::max(maxVal, v);
      sumVal += static_cast<long double>(v);
    }
    std::cout << label << " max=" << maxVal
              << " sum=" << static_cast<double>(sumVal) << "\n";
  };
  printFieldStats(implantConfig.labels.total);
  printFieldStats(implantConfig.labels.active);
  printFieldStats(implantConfig.labels.damage);
  printFieldStats(implantConfig.labels.interstitial);
  printFieldStats(implantConfig.labels.vacancy);
  std::cout << "-------------------------\n";

  std::cout << "\nDone.\n";
  std::cout << "  initial.vtu              : geometry + material IDs\n";
  std::cout << "  post_implant.vtu         : " << implantConfig.labels.total
            << " + " << implantConfig.labels.damage << " fields\n";
  std::cout << "  post_anneal.vtu          : " << implantConfig.labels.total
            << " + " << implantConfig.labels.active << " + I/V fields\n";
  std::cout << "  profile_post_implant.csv : depth profile after implant\n";
  std::cout << "  profile_post_anneal.csv  : depth profile after anneal\n";
  return 0;
}
