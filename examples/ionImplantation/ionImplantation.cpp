// Masked ion implantation + anneal — explicit dual-Pearson IV moments.
//
// Reads geometry and Pearson IV moments from a config file, builds a 2D
// masked-substrate domain, runs psIonImplantation with a dual-Pearson IV
// dopant profile and a table-driven damage model, then anneals with all
// physics (diffusivity, I/V, solid solubility) loaded from the vsclib CSV.
//
// Usage: ionImplantation [config.txt]
//
// Output VTU files (open in ParaView):
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
#include <fstream>
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

static std::unordered_map<std::string, std::string>
readRawConfig(const std::string &path) {
  std::unordered_map<std::string, std::string> out;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    auto hashPos = line.find('#');
    if (hashPos != std::string::npos)
      line = line.substr(0, hashPos);
    auto eqPos = line.find('=');
    if (eqPos == std::string::npos)
      continue;
    auto trim = [](std::string s) {
      const auto first = s.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
        return std::string{};
      const auto last = s.find_last_not_of(" \t\r\n");
      return s.substr(first, last - first + 1);
    };
    auto key = trim(line.substr(0, eqPos));
    auto val = trim(line.substr(eqPos + 1));
    if (!key.empty())
      out[key] = val;
  }
  return out;
}

int main(int argc, char *argv[]) {
  #ifdef VSCLIB_DIR
    viennaps::setVsclibRoot(VSCLIB_DIR);
  #endif
  using T = double;
  constexpr int D = 2;

  const std::string cfgPath = argc > 1 ? argv[1] : "config.txt";

  util::Parameters params;
  params.readConfigFile(cfgPath);
  const auto rawParams = readRawConfig(cfgPath);
  if (params.m.empty()) {
    std::cerr << "Config not found: " << cfgPath << "\n";
    std::cerr << "Usage: " << argv[0] << " [config.txt]\n";
    return 1;
  }
  std::cout << "--- ViennaPS Explicit Implant & Anneal (config: " << cfgPath
            << ") ---\n";

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

  // ── Build domain with the same y-bounds as the ViennaCS example ──────────
  //   y ∈ [−substrateDepth, topSpace + oxideThickness + maskHeight]
  //   x: REFLECTIVE boundary (symmetric about 0)
  //   y: INFINITE boundary (no implicit wrap)
  T bounds[2 * D] = {-0.5 * xExtent, 0.5 * xExtent, -substrateDepth,
                     topSpace + oxideThickness + maskHeight};
  BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                        BoundaryType::INFINITE_BOUNDARY};

  auto domain = Domain<T, D>::New(bounds, bc, gridDelta);

  // Helper: new ViennaLS level set sharing the same grid
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
  // Level set 1: Si substrate top (surface at y = 0)
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Si);
  }
  // Level set 2: screen oxide (y = 0 to y = oxideThickness)
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
  // Level set 3: hard mask (y = oxideThickness to y = oxideThickness + maskHeight)
  //              with a window of width openingWidth centred at x = 0
  {
    auto ls = makels();
    T origin[D] = {}, normal[D] = {};
    origin[D - 1] = oxideThickness + maskHeight;
    normal[D - 1] = 1.;
    viennals::MakeGeometry<T, D>(ls,
                                 viennals::Plane<T, D>::New(origin, normal))
        .apply();
    domain->insertNextLevelSetAsMaterial(ls, Material::Mask);

    // Cut the mask opening
    auto window = makels();
    T wMin[D] = {-0.5 * openingWidth, oxideThickness - gridDelta};
    T wMax[D] = {0.5 * openingWidth, oxideThickness + maskHeight + gridDelta};
    viennals::MakeGeometry<T, D>(window,
                                 viennals::Box<T, D>::New(wMin, wMax))
        .apply();
    domain->applyBooleanOperation(
        window, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
  }

  // Generate cell set: topSpace nm above the mask top, Air as cover material
  domain->generateCellSet(topSpace, Material::Air, /*isAboveSurface=*/true);
  domain->getCellSet()->buildNeighborhood();
  domain->getCellSet()->writeVTU("initial.vtu");

  // ── Implant ───────────────────────────────────────────────────────────────
  const std::string species =
      params.m.count("species") ? params.m.at("species") : "B";
  const std::string material =
      params.m.count("material") ? params.m.at("material") : "Si";
  const T tiltDeg   = params.m.count("angle") ? params.get("angle") : T(7);
  const T doseCm2   = params.get("doseCm2");
  const T energyKeV = params.get("energyKeV");

  PearsonIVParameters<T> headP;
  headP.mu    = params.get("projectedRange");
  headP.sigma = params.get("depthSigma");
  headP.beta  = params.get("skewness");
  headP.gamma = params.get("kurtosis");

  const T latMu    = params.m.count("lateralMu")
                         ? params.get("lateralMu")    : T(0);
  const T latSigma = params.m.count("lateralSigma")
                         ? params.get("lateralSigma") : T(5);

  SmartPointer<ImplantProfileModel<T, D>> implantModel;

  if (params.m.count("headFraction")) {
    PearsonIVParameters<T> tailP;
    tailP.mu    = params.m.count("tailProjectedRange")
                      ? params.get("tailProjectedRange") : headP.mu * T(2.5);
    tailP.sigma = params.m.count("tailDepthSigma")
                      ? params.get("tailDepthSigma") : headP.sigma * T(2.5);
    tailP.beta  = params.m.count("tailSkewness")
                      ? params.get("tailSkewness")  : T(0);
    tailP.gamma = params.m.count("tailKurtosis")
                      ? params.get("tailKurtosis")  : T(3);

    const T tailLatMu    = params.m.count("tailLateralMu")
                               ? params.get("tailLateralMu")    : T(0);
    const T tailLatSigma = params.m.count("tailLateralSigma")
                               ? params.get("tailLateralSigma") : latSigma;
    const T headFraction = params.get("headFraction");

    std::cout << "Implanting " << species << " into " << material
              << " at " << energyKeV << " keV, " << tiltDeg
              << " deg tilt (dual-Pearson IV, head fraction "
              << headFraction << ") ...\n";

    implantModel = SmartPointer<ImplantDualPearsonIV<T, D>>::New(
        headP, tailP, headFraction, latMu, latSigma, tailLatMu, tailLatSigma);
  } else {
    std::cout << "Implanting " << species << " into " << material
              << " at " << energyKeV << " keV, " << tiltDeg
              << " deg tilt (Pearson IV) ...\n";
    implantModel =
        SmartPointer<ImplantPearsonIV<T, D>>::New(headP, latMu, latSigma);
  }

  const T rotationDeg =
      params.m.count("rotationDeg") ? params.get("rotationDeg") : T(0);
  DamageRecipe<T> damageRecipe;
  damageRecipe.species         = species;
  damageRecipe.material        = material;
  damageRecipe.energyKeV       = energyKeV;
  damageRecipe.tiltDeg         = tiltDeg;
  damageRecipe.rotationDeg     = rotationDeg;
  damageRecipe.dosePerCm2      = doseCm2;
  damageRecipe.screenThickness = screenThickness;

  auto implant = SmartPointer<IonImplantation<T, D>>::New();
  implant->setImplantModel(implantModel);
  implant->setDamageModel(
      SmartPointer<RecipeDrivenDamageModel<T, D>>::New(damageRecipe));
  implant->setTiltAngle(tiltDeg);
  implant->setDose(doseCm2);
  implant->setLengthUnit(T(1e-7));  // Mandotary: scales dose from cm^-2 to nm^-2
  implant->setDoseControl(ImplantDoseControl::WaferDose);
  implant->setMaskMaterials({Material::Mask});
  implant->setScreenMaterials({Material::SiO2});

  Process<T, D>(domain, implant, T(0)).apply();
  domain->getCellSet()->writeVTU("post_implant.vtu");

  // ── Anneal ────────────────────────────────────────────────────────────────
  std::vector<double> durations, temps;
  if (rawParams.count("annealStepDurations"))
    durations = parseDoubleList(rawParams.at("annealStepDurations"));
  if (rawParams.count("annealTemperatures"))
    temps = parseDoubleList(rawParams.at("annealTemperatures"));

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
  const std::string substrateMaterial = "silicon";

  std::cout << "\n--- ANNEAL PARAMETERS (C++) ---\n";
  std::cout << "Dopant Name:         " << dopantName << "\n";
  std::cout << "Peak Temperature:    " << peakT << " K\n";
  
  std::cout << "Durations:           [";
  for (size_t i = 0; i < durations.size(); ++i) std::cout << durations[i] << (i + 1 < durations.size() ? ", " : "");
  std::cout << "]\n";
  
  std::cout << "Temperatures:        [";
  for (size_t i = 0; i < temps.size(); ++i) std::cout << temps[i] << (i + 1 < temps.size() ? ", " : "");
  std::cout << "]\n";

  std::cout << "Defect Coupling:     " << (!params.m.count("annealDefectCoupling") || params.get("annealDefectCoupling") != T(0) ? "true" : "false") << "\n";
  std::cout << "Anneal Mode:         " << (params.m.count("annealMode") ? params.m.at("annealMode") : "implicit (default)") << "\n";
  std::cout << "Diffusion Materials: {Material::Si (10)}\n";
  std::cout << "Blocking Materials:  {Material::Mask (0), Material::SiO2 (11)}\n";

  auto cs = domain->getCellSet();
  if (cs && cs->getScalarData("concentration")) {
    auto conc = cs->getScalarData("concentration");
    T maxC = 0.0;
    for (T val : *conc) maxC = std::max(maxC, val);
    std::cout << "Max Concentration:   " << maxC << " (entering anneal)\n";

    if (cs->getScalarData("Damage")) {
      auto dmg = cs->getScalarData("Damage");
      T maxD = 0.0;
      for (T val : *dmg) maxD = std::max(maxD, val);
      std::cout << "Max Damage:          " << maxD << " (entering anneal)\n";
    }
  }
  std::cout << "-------------------------------\n\n";

  std::cout << "\n--- ANNEALING MODEL PARAMETERS (from vsclib CSV) ---\n";
  #ifdef VSCLIB_DIR
    std::string csvPath = std::string(VSCLIB_DIR) + "/anneal/annealing.csv";
  #else
    std::string csvPath = viennaps::getVsclibRoot() + "/anneal/annealing.csv";
  #endif
  std::cout << "vsclib root: " << viennaps::getVsclibRoot() << "\n";
  std::cout << "anneal CSV:  " << csvPath << "\n";
  std::cout << "dopant:      " << dopantName << "\n";
  std::cout << "substrate:   " << substrateMaterial << "\n";
  std::ifstream file(csvPath);
  if (file.is_open()) {
    std::string line;
    std::vector<std::string> headers;
    
    // Find header line (skip empty lines and comments)
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream hs(line);
      std::string h;
      while (std::getline(hs, h, ',')) headers.push_back(h);
      break;
    }

    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::vector<std::string> vals;
      std::stringstream ls(line);
      std::string v;
      while (std::getline(ls, v, ',')) vals.push_back(v);

      std::string lowerLine = line;
      std::transform(lowerLine.begin(), lowerLine.end(), lowerLine.begin(), ::tolower);
      std::string lowerDopant = dopantName;
      std::transform(lowerDopant.begin(), lowerDopant.end(), lowerDopant.begin(), ::tolower);

      if (lowerLine.find(lowerDopant) != std::string::npos && lowerLine.find("silicon") != std::string::npos) {
        double D0 = 0.0, Ea = 0.0;
        bool isMainDiffusivity = false;
        for (size_t i = 0; i < headers.size() && i < vals.size(); ++i) {
          if (!vals[i].empty()) {
            if (headers[i] == "D0_cm2_per_s") { D0 = std::stod(vals[i]); isMainDiffusivity = true; }
            if (headers[i] == "Ea_eV") Ea = std::stod(vals[i]);
          }
        }
        if (isMainDiffusivity) {
          std::cout << "Found parameters for " << dopantName << " in silicon:\n";
          for (size_t i = 0; i < headers.size() && i < vals.size(); ++i) {
            if (!vals[i].empty()) std::cout << "  " << headers[i] << ": " << vals[i] << "\n";
          }
          const double kB = 8.617333262145e-5;
          double DT = D0 * std::exp(-Ea / (kB * peakT));
          std::cout << "  -> Evaluated Diffusivity at " << peakT << " K: " << DT << " cm^2/s\n";
          break;
        }
      }
    }
  } else {
    std::cout << "Could not open CSV at: " << csvPath << "\n";
  }
  std::cout << "----------------------------------------------------\n\n";

  auto anneal = SmartPointer<Anneal<T, D>>::New();
  anneal->setSpeciesLabel("concentration");
  anneal->setDopantName(dopantName);
  anneal->setTemperature(peakT);
  anneal->loadAnnealingCSV();  // loads D0/Ea, I/V params, solid solubility

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
  domain->getCellSet()->writeVTU("post_anneal.vtu");

  std::cout << "\n--- POST-ANNEAL STATS (ViennaPS C++) ---\n";
  auto printFieldStats = [&](const std::string &label) {
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
    std::cout << label << " max=" << maxVal << " sum=" << static_cast<double>(sumVal)
              << "\n";
  };
  printFieldStats("concentration");
  printFieldStats("active_concentration");
  printFieldStats("Damage");
  printFieldStats("Interstitial");
  printFieldStats("Vacancy");
  std::cout << "----------------------------------------\n";

  std::cout << "Done.\n";
  std::cout << "  initial.vtu      : geometry + material IDs\n";
  std::cout << "  post_implant.vtu : concentration + Damage fields\n";
  std::cout << "  post_anneal.vtu  : concentration + active_concentration"
               " + I/V fields\n";
  return 0;
}
