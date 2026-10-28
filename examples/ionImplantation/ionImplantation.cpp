// Masked ion implantation + anneal — unified example.
//
// Reads geometry and process parameters from a config file, builds a 2D
// masked-substrate domain, and executes an implant + anneal pipeline.
//
// The example automatically switches modes based on the configuration file:
// - If 'projectedRange' is present (e.g., config.txt): uses explicit
//   Pearson IV moments and a manually parameterized anneal model.
// - If omitted (e.g., config_default.txt): uses the table-driven ViennaPS
//   model database for both implant moments and anneal physics.
//
// Usage: ionImplantation [config.txt | config_default.txt]
//
// Output VTU files (open in ParaView):
//   initial.vtu      — geometry + material IDs
//   post_implant.vtu — dopant total concentration + damage fields
//   post_anneal.vtu  — dopant total/active concentration + I/V fields

#include "exampleConfig.hpp"
#include <geometries/psMakePlane.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

using namespace viennaps;

static int runIonImplantation(int argc, char *argv[]) {
  using T = double;
  constexpr int D = 2;

  const std::string cfgPath = argc > 1 ? argv[1] : "config.txt";

  util::Parameters params;
  params.readConfigFile(cfgPath);
  if (params.m.empty()) {
    std::cerr << "Config not found: " << cfgPath << "\n";
    std::cerr << "Usage: " << argv[0] << " [config.txt]\n";
    return 1;
  }

  // Determine mode: if 'projectedRange' is omitted, use the table-driven DB
  // defaults.
  const bool useTable = params.contains("projectedRange");

  if (useTable) {
    initModelDbRoot();
  }

  std::cout << "--- ViennaPS " << (useTable ? "Table-Driven" : "Explicit")
            << " Implant & Anneal (config: " << cfgPath << ") ---\n";

  // ── Geometry parameters ───────────────────────────────────────────────────
  const T gridDelta = params.get("gridDelta");
  const T xExtent = params.get("xExtent");
  const T topSpace = params.get("topSpace");
  const T substrateDepth = params.get("substrateDepth");
  const T openingWidth = params.get("openingWidth");
  const T maskHeight = params.get("maskHeight");
  const T oxideThickness = params.m.count("screenOxideThickness")
                               ? params.get("screenOxideThickness")
                               : params.get("oxideThickness");
  const T screenThickness = params.m.count("screenThickness")
                                ? params.get("screenThickness")
                                : oxideThickness;

  // ── Build domain with the same y-bounds as the ViennaCS example ──────────
  //   y ∈ [−substrateDepth, topSpace + oxideThickness + maskHeight]
  //   x: REFLECTIVE boundary (symmetric about 0)
  //   y: INFINITE boundary (no implicit wrap)
  T bounds[2 * D] = {-0.5 * xExtent, 0.5 * xExtent, -substrateDepth,
                     topSpace + oxideThickness + maskHeight};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  auto domain = Domain<T, D>::New(bounds, bc, gridDelta);

  // Level set 0: Si substrate bottom
  MakePlane<T, D>(domain, -substrateDepth, Material::Si).apply();
  // Level set 1: Si substrate top (surface at y = 0)
  MakePlane<T, D>(domain, 0.0, Material::Si, true).apply();
  // Level set 2: screen oxide (y = oxideThickness)
  MakePlane<T, D>(domain, oxideThickness, Material::SiO2, true).apply();
  // Level set 3: hard mask (y = oxideThickness + maskHeight)
  //              with a window of width openingWidth centred at x = 0
  MakePlane<T, D>(domain, oxideThickness + maskHeight, Material::Mask, true)
      .apply();
  {
    // Cut the mask opening
    auto window = viennals::Domain<T, D>::New(bounds, bc, gridDelta);
    T wMin[D] = {-0.5 * openingWidth, oxideThickness - gridDelta};
    T wMax[D] = {0.5 * openingWidth, oxideThickness + maskHeight + gridDelta};
    viennals::MakeGeometry<T, D>(window, viennals::Box<T, D>::New(wMin, wMax))
        .apply();
    domain->applyBooleanOperation(
        window, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  // Generate cell set: topSpace nm above the mask top, Air as cover material
  domain->generateCellSet(topSpace, Material::Air, /*isAboveSurface=*/true,
                          /*withEmbeddedBoundaries=*/true);
  domain->getCellSet()->buildNeighborhood();

  std::string outSuffix = useTable ? "_preset.vtu" : "_manual.vtu";
  domain->getCellSet()->writeVTU("initial" + outSuffix);

  // ── Setup Process Models ────────────────────────────────────────────────
  auto implant = SmartPointer<IonImplantation<T, D>>::New();
  auto anneal = SmartPointer<Anneal<T, D>>::New();

  const auto annealSchedule = ionimpl::readAnnealSchedule<T>(params);
  const T peakT = peakAnnealTemperature(annealSchedule);

  std::string labelTotal, labelActive, labelDamage, labelInterstitial,
      labelVacancy;

  if (useTable) {
    const auto implantConfig =
        ionimpl::makeTableImplantSetup<T, D>(params, screenThickness);
    std::cout << "Implanting " << implantConfig.description << " ...\n";
    applyImplantSetup(*implant, implantConfig);

    std::cout << "Annealing: peak T = " << (peakT - T(273.15)) << " C ...\n";
    const auto annealConfig = ionimpl::makeAnnealSetup<T>(
        params, annealSchedule, implantConfig, peakT, {Material::Si},
        {Material::Mask, Material::SiO2},
        /*defaultUseModelDb=*/true);
    std::cout << "Anneal parameter source: " << annealConfig.model.source
              << "\n";
    applyAnnealSetup(*anneal, annealConfig);

    labelTotal = implantConfig.labels.total;
    labelActive = implantConfig.labels.active;
    labelDamage = implantConfig.labels.damage;
    labelInterstitial = implantConfig.labels.interstitial;
    labelVacancy = implantConfig.labels.vacancy;
  } else {
    const auto implantConfig =
        ionimpl::makeAnalyticImplantSetup<T, D>(params, screenThickness);
    std::cout << "Implanting " << implantConfig.description << " ...\n";
    applyImplantSetup(*implant, implantConfig);

    std::cout << "Annealing: peak T = " << (peakT - T(273.15)) << " C ...\n";
    const auto annealConfig = ionimpl::makeAnnealSetup<T>(
        params, annealSchedule, implantConfig, peakT, {Material::Si},
        {Material::Mask, Material::SiO2},
        /*defaultUseModelDb=*/false);
    std::cout << "Anneal parameter source: " << annealConfig.model.source
              << "\n";
    applyAnnealSetup(*anneal, annealConfig);

    labelTotal = implantConfig.labels.total;
    labelActive = implantConfig.labels.active;
    labelDamage = implantConfig.labels.damage;
    labelInterstitial = implantConfig.labels.interstitial;
    labelVacancy = implantConfig.labels.vacancy;
  }

  // ── Run Processes ───────────────────────────────────────────────────────
  Process<T, D>(domain, implant, T(0)).apply();
  domain->getCellSet()->writeVTU("post_implant" + outSuffix);

  Process<T, D>(domain, anneal, T(0)).apply();
  domain->getCellSet()->writeVTU("post_anneal" + outSuffix);

  std::cout << "\n--- POST-ANNEAL STATS (ViennaPS C++) ---\n";
  auto printFieldStats = [&](const std::string &label) {
    if (label.empty())
      return;
    auto field = domain->getCellSet()->getScalarData(label);
    if (!field) {
      std::cout << label << " <missing>\n";
      return;
    }
    T maxVal = 0.;
    long double sumVal = 0.;
    for (const auto &v : *field) {
      maxVal = std::max(maxVal, v);
      sumVal += static_cast<long double>(v);
    }
    std::cout << label << " max=" << maxVal
              << " sum=" << static_cast<double>(sumVal) << "\n";
  };
  printFieldStats(labelTotal);
  printFieldStats(labelActive);
  printFieldStats(labelDamage);
  printFieldStats(labelInterstitial);
  printFieldStats(labelVacancy);
  std::cout << "----------------------------------------\n";

  std::cout << "Done.\n";
  std::cout << "  initial" << outSuffix << "      : geometry + material IDs\n";
  std::cout << "  post_implant" << outSuffix << " : " << labelTotal << " + "
            << labelDamage << " fields\n";
  std::cout << "  post_anneal" << outSuffix << "  : " << labelTotal << " + "
            << labelActive << " + I/V fields\n";
  return 0;
}

int main(int argc, char *argv[]) {
  return viennaps::modeldb::runWithModelDbErrors(
      [&]() { return runIonImplantation(argc, argv); });
}
