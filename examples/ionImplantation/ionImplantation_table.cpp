// Masked ion implantation + anneal — table-driven (vsclib database).
//
// All implant moments are looked up from the internal vsclib table using
// (species, material, substrateType, energyKeV, tilt, rotation, dose,
// screenThickness).  Anneal physics are loaded from the same vsclib CSV.
//
// Usage: ionImplantation_table [config_table.txt]
//
// Output VTU files:
//   initial.vtu      — geometry + material IDs
//   post_implant.vtu — concentration + Damage fields
//   post_anneal.vtu  — concentration + active_concentration + I/V fields

#include <models/psAnneal.hpp>
#include <models/psIonImplantation.hpp>
#include <psDomain.hpp>
#include <process/psProcess.hpp>

#include <lsBooleanOperation.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>

#include <vcUtil.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace viennaps;

// ── Helper: parse a comma-separated list of doubles ───────────────────────────
static std::vector<double> parseDoubleList(const std::string &s) {
  std::vector<double> result;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    try {
      result.push_back(std::stod(token));
    } catch (...) {
    }
  }
  return result;
}

int main(int argc, char *argv[]) {
  #ifdef VSCLIB_DIR
    viennaps::setVsclibRoot(VSCLIB_DIR);
  #endif
  using T = double;
  constexpr int D = 2;

  const std::string cfgPath = argc > 1 ? argv[1] : "config_table.txt";

  util::Parameters params;
  params.readConfigFile(cfgPath);
  if (params.m.empty()) {
    std::cerr << "Config not found: " << cfgPath << "\n";
    std::cerr << "Usage: " << argv[0] << " [config_table.txt]\n";
    return 1;
  }
  std::cout << "--- ViennaPS Table-Driven Implant & Anneal (config: "
            << cfgPath << ") ---\n";

  // ── Geometry parameters ───────────────────────────────────────────────────
  const T gridDelta      = params.get("gridDelta");
  const T xExtent        = params.get("xExtent");
  const T topSpace       = params.get("topSpace");
  const T substrateDepth = params.get("substrateDepth");
  const T openingWidth   = params.get("openingWidth");
  const T maskHeight     = params.get("maskHeight");
  const T oxideThickness =
      params.m.count("screenOxideThickness")
          ? params.get("screenOxideThickness")
          : params.get("oxideThickness");
  const T screenThickness =
      params.m.count("screenThickness") ? params.get("screenThickness")
                                        : oxideThickness;

  // ── Build domain ─────────────────────────────────────────────────────────
  T bounds[2 * D] = {-0.5 * xExtent, 0.5 * xExtent, -substrateDepth,
                     topSpace + oxideThickness + maskHeight};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  auto domain = Domain<T, D>::New(bounds, bc, gridDelta);

  auto makels = [&]() {
    return SmartPointer<viennals::Domain<T, D>>::New(bounds, bc, gridDelta);
  };

  // Level set 0: Si substrate bottom
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = -substrateDepth;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  // Level set 1: Si substrate top
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  // Level set 2: screen oxide
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = oxideThickness;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::SiO2);
  }
  // Level set 3: hard mask with window
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = oxideThickness + maskHeight;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);

    auto window = makels();
    T wMin[D] = {-0.5 * openingWidth, oxideThickness - gridDelta};
    T wMax[D] = {0.5 * openingWidth, oxideThickness + maskHeight + gridDelta};
    viennals::MakeGeometry<T, D>(window,
                                 viennals::Box<T, D>::New(wMin, wMax))
        .apply();
    domain->applyBooleanOperation(
        window, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  domain->generateCellSet(topSpace, Material::Air, /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  domain->getCellSet()->writeVTU("initial_table.vtu");

  // ── Implant (table-driven) ─────────────────────────────────────────────────
  const std::string species =
      params.m.count("species") ? params.m.at("species") : "B";
  const std::string material =
      params.m.count("material") ? params.m.at("material") : "Si";
  const std::string substrateType =
      params.m.count("substrateType") ? params.m.at("substrateType")
                                      : "crystalline";
  const T energyKeV   = params.get("energyKeV");
  const T tiltDeg     = params.m.count("angle") ? params.get("angle") : T(7);
  const T doseCm2     = params.get("doseCm2");
  const T rotationDeg = params.m.count("rotationDeg")
                            ? params.get("rotationDeg") : T(0);

  std::cout << "Implanting " << species << " into " << material
            << " at " << energyKeV << " keV, " << tiltDeg
            << " deg tilt (table lookup) ...\n";

  ImplantRecipe<T> recipe;
  recipe.species         = species;
  recipe.material        = material;
  recipe.substrateType   = substrateType;
  recipe.energyKeV       = energyKeV;
  recipe.tiltDeg         = tiltDeg;
  recipe.rotationDeg     = rotationDeg;
  recipe.dosePerCm2      = doseCm2;
  recipe.screenThickness = screenThickness;

  DamageRecipe<T> dmgRecipe;
  dmgRecipe.species         = species;
  dmgRecipe.material        = material;
  dmgRecipe.energyKeV       = energyKeV;
  dmgRecipe.tiltDeg         = tiltDeg;
  dmgRecipe.rotationDeg     = rotationDeg;
  dmgRecipe.dosePerCm2      = doseCm2;
  dmgRecipe.screenThickness = screenThickness;

  auto implant = SmartPointer<IonImplantation<T, D>>::New();
  implant->setImplantModel(
      SmartPointer<RecipeDrivenImplantModel<T, D>>::New(recipe));
  implant->setDamageModel(
      SmartPointer<RecipeDrivenDamageModel<T, D>>::New(dmgRecipe));
  implant->setTiltAngle(tiltDeg);
  implant->setDose(doseCm2);
  implant->setLengthUnit(T(1e-7));  // Mandotary: scales dose from cm^-2 to nm^-2
  implant->setDoseControl(ImplantDoseControl::WaferDose);
  implant->setMaskMaterials({Material::Mask});
  implant->setScreenMaterials({Material::SiO2});

  Process<T, D>(domain, implant, T(0)).apply();
  domain->getCellSet()->writeVTU("post_implant_table.vtu");

  // ── Anneal ────────────────────────────────────────────────────────────────
  std::vector<double> durations, temps;
  if (params.m.count("annealStepDurations"))
    durations = parseDoubleList(params.m.at("annealStepDurations"));
  if (params.m.count("annealTemperatures"))
    temps = parseDoubleList(params.m.at("annealTemperatures"));

  const T peakT = temps.empty()
                      ? T(1323.15)
                      : *std::max_element(temps.begin(), temps.end());
  std::cout << "Annealing: peak T = " << (peakT - T(273.15)) << " C ...\n";

  static const std::unordered_map<std::string, std::string> kDopantNames = {
      {"B", "boron"},   {"As", "arsenic"},  {"P", "phosphorus"},
      {"Sb", "antimony"}, {"In", "indium"}, {"C", "carbon"},
      {"F", "fluorine"}, {"N", "nitrogen"}, {"Al", "aluminum"},
  };
  const std::string dopantName =
      kDopantNames.count(species) ? kDopantNames.at(species) : species;

  auto anneal = SmartPointer<Anneal<T, D>>::New();
  anneal->setSpeciesLabel("concentration");
  anneal->setDopantName(dopantName);
  anneal->setTemperature(peakT);
  anneal->loadAnnealingCSV();

  // Set solver mode (implicit is unconditionally stable for long anneals)
  if (params.m.count("annealMode")) {
    std::string mode = params.m.at("annealMode");
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    if (mode == "implicit") {
      anneal->setMode(AnnealMode::GaussSeidel);
    } else if (mode == "explicit") {
      anneal->setMode(AnnealMode::Explicit);
    }
  }

  if (!durations.empty() && !temps.empty()) {
    anneal->setTemperatureSchedule(
        std::vector<T>(durations.begin(), durations.end()),
        std::vector<T>(temps.begin(), temps.end()));
  } else {
    anneal->setDuration(params.m.count("annealDuration")
                            ? T(params.get("annealDuration")) : T(5));
  }
  anneal->setDiffusionMaterials({Material::Si});
  anneal->setBlockingMaterials({Material::Mask, Material::SiO2});

  const bool defectCoupling =
      !params.m.count("annealDefectCoupling") ||
      params.get("annealDefectCoupling") != T(0);
  if (defectCoupling)
    anneal->enableDefectCoupling(true);

  Process<T, D>(domain, anneal, T(0)).apply();
  domain->getCellSet()->writeVTU("post_anneal_table.vtu");

  std::cout << "Done.\n";
  std::cout << "  initial_table.vtu      : geometry + material IDs\n";
  std::cout << "  post_implant_table.vtu : concentration + Damage fields\n";
  std::cout << "  post_anneal_table.vtu  : concentration + active_concentration"
               " + I/V fields\n";
  return 0;
}
