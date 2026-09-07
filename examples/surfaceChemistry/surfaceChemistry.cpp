// One model, many chemistries.
//
// Every mechanism in `reactions/` runs through THIS program. Nothing about a
// particular chemistry is written here: the reaction file decides whether the
// surface grows or is etched, which particles are traced, how many coverages
// there are, what the rate laws look like, and how the chemistry differs from
// one material to the next.
//
//     ./surfaceChemistry reactions/silane.mechanism.json        # deposition
//     ./surfaceChemistry reactions/sf6o2.mechanism.json         # etching
//     ./surfaceChemistry reactions/polymer_etch.mechanism.json  # both at once
//
// The `.mechanism.json` beside each `.yaml` is the same reactions in machine
// form, compiled by ViennaChem:
//
//     python -m viennachem reactions/silane.yaml reactions/silane.mechanism.json
//
// C++ reads it directly, so this program needs no Python. See README.md for
// what each reaction file demonstrates.

#include <geometries/psGeometryFactory.hpp>
#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psChemicalMechanismIO.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>


#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using namespace viennaps;

namespace {

using NumericType = double;

struct Options {
  std::string reactions = "reactions/silane.mechanism.json";
  int dim = 2;
  NumericType thickness = 20.; // nm of film grown, or removed
  NumericType gridDelta = 2.;
  NumericType width = 80.;  // trench width or hole diameter
  NumericType depth = 120.; // trench or hole depth
  NumericType mask = 0.;    // mask height; an etch needs one to be selective
  int rays = 1000;
  bool gpu = false;
  // A fin instead of a trench, which puts a convex corner in front of the
  // chemistry rather than a concave one.
  bool fin = false;
  // A superlattice instead of a single substrate. `stack` counts the Si/SiGe
  // pairs, and tapered trenches cut through all of them leave a pillar whose
  // sidewall exposes every layer in turn under an oxide cap.
  int stack = 0;
  NumericType layer = 12.;  // nm per layer
  NumericType oxide = 15.;  // nm of SiO2 above the top layer
  NumericType taper = 3.;   // degrees, sidewall taper of the trenches
  NumericType extent = 200.;    // nm, lateral domain width
  NumericType height = 0.;      // nm, vertical extent; 0 sizes it from the run
  NumericType time = 0.;        // s, 0 derives it from thickness and the rate
  // Overrides applied to the compiled mechanism before anything is solved.
  NumericType temperature = 0.;                        // K, 0 keeps the file's
  std::vector<std::pair<std::string, NumericType>> fluxes;
  // Scale factors on a species' sticking, which the rate law and the ray
  // termination both read, so one factor moves the chemistry and the transport
  // together. This is the k_ads of the file expressed as a multiple.
  std::vector<std::pair<std::string, NumericType>> stickings;
  // Run the hand-written ViennaPS model of the same chemistry instead of the
  // reaction file, so the two can be put in the same geometry.
  bool handwritten = false;
  // Prefactor of one reaction, by the index the run prints it at. A reverse
  // constant is a reaction like any other once the file has been compiled, so
  // this is how one is varied without editing the file.
  std::vector<std::pair<int, NumericType>> rates;
  // Mean ion energy, which Eq. (4) reads through every yield at once.
  NumericType ionEnergy = 0.;
  bool intermediate = false;  // write the flux and coverage fields
  // Time the coverage solve alone, over this many surface points, with the
  // transport left out entirely.
  int bench = 0;
  int benchRepeats = 20;
  bool profile = false;  // time the coverage solve inside a full run
  // The material label given to the deposited film. It defaults to the first
  // solid the mechanism declares, which is not always a material name.
  std::string film;
  // The material the substrate is made of, which a mechanism's per-material
  // blocks are resolved against. An oxide etch has to be run on an oxide.
  std::string substrate = "Si";
  // Surfaces written during the process rather than only at its end.
  int snapshots = 1;
  std::string out;
};

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value after " << arg << "\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--gpu")
      o.gpu = true;
    else if (arg == "-D" || arg == "--dim")
      o.dim = std::stoi(next());
    else if (arg == "--thickness")
      o.thickness = std::stod(next());
    else if (arg == "--grid")
      o.gridDelta = std::stod(next());
    else if (arg == "--width")
      o.width = std::stod(next());
    else if (arg == "--depth")
      o.depth = std::stod(next());
    else if (arg == "--mask")
      o.mask = std::stod(next());
    else if (arg == "--fin")
      o.fin = true;
    else if (arg == "--stack")
      o.stack = std::stoi(next());
    else if (arg == "--layer")
      o.layer = std::stod(next());
    else if (arg == "--oxide")
      o.oxide = std::stod(next());
    else if (arg == "--taper")
      o.taper = std::stod(next());
    else if (arg == "--sticking") {
      const std::string spec = next();
      const auto eq = spec.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--sticking wants LABEL=FACTOR, got " << spec << "\n";
        std::exit(1);
      }
      o.stickings.emplace_back(spec.substr(0, eq),
                               std::stod(spec.substr(eq + 1)));
    }
    else if (arg == "--handwritten")
      o.handwritten = true;
    else if (arg == "--ion-energy")
      o.ionEnergy = std::stod(next());
    else if (arg == "--rate") {
      const std::string spec = next();
      const auto eq = spec.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--rate wants INDEX=VALUE, got " << spec << "\n";
        std::exit(1);
      }
      o.rates.emplace_back(std::stoi(spec.substr(0, eq)),
                           std::stod(spec.substr(eq + 1)));
    }
    else if (arg == "--intermediate")
      o.intermediate = true;
    else if (arg == "--profile")
      o.profile = true;
    else if (arg == "--bench")
      o.bench = std::stoi(next());
    else if (arg == "--bench-repeats")
      o.benchRepeats = std::stoi(next());
    else if (arg == "--extent")
      o.extent = std::stod(next());
    else if (arg == "--height")
      o.height = std::stod(next());
    else if (arg == "--time")
      o.time = std::stod(next());
    else if (arg == "--temperature")
      o.temperature = std::stod(next());
    else if (arg == "--flux") {
      const std::string spec = next();
      const auto eq = spec.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--flux wants LABEL=VALUE, got " << spec << "\n";
        std::exit(1);
      }
      o.fluxes.emplace_back(spec.substr(0, eq), std::stod(spec.substr(eq + 1)));
    }
    else if (arg == "--snapshots")
      o.snapshots = std::stoi(next());
    else if (arg == "--substrate")
      o.substrate = next();
    else if (arg == "--film")
      o.film = next();
    else if (arg == "--out")
      o.out = next();
    else if (arg == "--rays")
      o.rays = std::stoi(next());
    else if (arg == "-r" || arg == "--reactions")
      o.reactions = next();
    else if (arg.rfind("-", 0) != 0)
      o.reactions = arg; // the file may be given positionally
    else {
      std::cerr << "usage: surfaceChemistry [reactions.mechanism.json] [-D 2|3] "
                   "[--thickness nm] [--grid nm] [--width nm] [--depth nm] "
                   "[--mask nm] [--rays n] [--gpu]\n";
      std::exit(1);
    }
  }
  return o;
}

// What the model derived from the file, before anything is simulated.
void report(const ChemicalMechanism<NumericType> &mech,
            const std::vector<NumericType> &theta, NumericType rate) {
  std::cout << "mechanism   : " << mech.name << "\n"
            << "temperature : " << mech.temperature << " K\n";
  std::cout << "solids      :";
  for (const auto &s : mech.solids)
    std::cout << "  " << s.name << " (rho = " << s.rho << " e22/cm3)";
  std::cout << "\ncoverages   :";
  for (const auto &c : mech.coverageNames)
    std::cout << "  " << c;
  std::cout << "\nparticles   :";
  for (const auto &g : mech.gas)
    if (g.traced && !g.isIonChannel)
      std::cout << "  " << g.label;
  std::cout << "\nsticking    :";
  for (size_t i = 0; i < mech.gas.size(); ++i)
    if (mech.gas[i].traced && !mech.gas[i].isIonChannel)
      std::cout << "  " << mech.gas[i].label << " = " << std::scientific
                << std::setprecision(3) << mech.stickingOf(int(i))
                << std::defaultfloat;
  if (mech.ionSource.present)
    std::cout << "  [ion, " << mech.ionSource.meanEnergy << " eV]";
  std::cout << "\nreactions   :\n";
  for (size_t j = 0; j < mech.reactions.size(); ++j)
    std::cout << "  [" << j << "] " << mech.reactions[j].equation
              << "   (prefactor " << mech.reactions[j].prefactor << ")\n";
  std::cout << "steady state:\n";
  for (size_t i = 0; i < theta.size(); ++i)
    std::cout << "   theta_" << mech.coverageNames[i] << " = " << std::scientific
              << std::setprecision(6) << theta[i] << "\n";
  std::cout << "   " << (rate < 0. ? "etch rate  " : "growth rate")
            << " = " << rate << " nm/s\n"
            << std::defaultfloat;
}

// Forwards every call to the model it wraps and times the coverage solve, so
// that the cost of finding the coverages can be read as a fraction of the run
// it sits inside rather than on its own.
template <typename NumericType, int D>
class TimedSurfaceModel : public SurfaceModel<NumericType> {
  SmartPointer<SurfaceModel<NumericType>> inner_;

public:
  double seconds = 0.;
  long calls = 0;

  explicit TimedSurfaceModel(SmartPointer<SurfaceModel<NumericType>> inner)
      : inner_(std::move(inner)) {}

  void initializeCoverages(unsigned n) override {
    inner_->initializeCoverages(n);
    this->coverages = inner_->getCoverages();
  }
  void initializeCoverages(unsigned n,
                           const std::vector<Vec3D<NumericType>> &c) override {
    inner_->initializeCoverages(n, c);
    this->coverages = inner_->getCoverages();
  }
  void initializeSurfaceData(unsigned n) override {
    inner_->initializeSurfaceData(n);
    this->surfaceData = inner_->getSurfaceData();
  }
  void initializeProcessParameters() override {
    inner_->initializeProcessParameters();
  }
  void setSurfaceCoordinates(const std::vector<Vec3D<NumericType>> &c) override {
    inner_->setSurfaceCoordinates(c);
  }
  void setTimeStep(NumericType dt) override { inner_->setTimeStep(dt); }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coords,
                      const std::vector<NumericType> &materialIds) override {
    return inner_->calculateVelocities(fluxes, coords, materialIds);
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    const auto t0 = std::chrono::steady_clock::now();
    inner_->updateCoverages(fluxes, materialIds);
    seconds += std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
    ++calls;
    this->coverages = inner_->getCoverages();
  }
};

template <int D> int run(const Options &o) {
  const auto wallStart = std::chrono::steady_clock::now();
  units::Length::setUnit("nm");
  units::Time::setUnit("s");

  auto mech = readChemicalMechanism<NumericType>(o.reactions);

  // A condition the file declares can be overridden on the command line, which
  // is how a sweep over one number is run without writing a file per point.
  if (o.temperature > 0.)
    mech.temperature = o.temperature;
  for (const auto &[label, value] : o.fluxes) {
    bool found = false;
    for (auto &g : mech.gas) {
      // "ion" reaches every yield channel at once, which is where the ion
      // source flux lives once the yields have been folded into it.
      const bool hit = label == "ion" ? g.isIonChannel
                                      : (g.label == label ||
                                         g.label == label + "_flux");
      if (hit) {
        g.sourceFlux = value;
        found = true;
      }
    }
    if (!found) {
      std::cerr << "--flux: '" << label << "' is not a traced species of "
                << mech.name << "\n";
      return 1;
    }
  }
  if (o.ionEnergy > 0.) {
    if (!mech.ionSource.present) {
      std::cerr << "--ion-energy: " << mech.name << " declares no ion\n";
      return 1;
    }
    // The width of the distribution is kept a fixed fraction of its mean, so
    // the two energies are compared at the same relative spread.
    mech.ionSource.sigmaEnergy *= o.ionEnergy / mech.ionSource.meanEnergy;
    mech.ionSource.meanEnergy = o.ionEnergy;
  }
  for (const auto &[index, value] : o.rates) {
    if (index < 0 || index >= static_cast<int>(mech.reactions.size())) {
      std::cerr << "--rate: reaction " << index << " is out of range\n";
      return 1;
    }
    auto &r = mech.reactions[index];
    r.prefactor = value;
    auto d = r.materialConstant.getDefault();
    d.prefactor = value;
    r.materialConstant.setDefault(d);
  }
  for (const auto &[label, factor] : o.stickings) {
    bool found = false;
    for (auto &g : mech.gas) {
      if (g.label != label && g.label != label + "_flux")
        continue;
      found = true;
      g.s0 *= factor;
      // The same number is held twice, once as the prefactor of the adsorption
      // step's rate law and once as the sticking the ray tracer terminates on,
      // so a change to it has to reach both or the surface solve and the
      // transport stop describing the same chemistry.
      const int gasIndex = static_cast<int>(&g - mech.gas.data());
      for (auto &r : mech.reactions) {
        if (!r.isAdsorption)
          continue;
        bool consumes = false;
        for (const auto &f : r.gasFactors)
          consumes |= f.index == gasIndex;
        if (!consumes)
          continue;
        r.prefactor *= factor;
        auto d = r.materialConstant.getDefault();
        d.prefactor *= factor;
        r.materialConstant.setDefault(d);
#define PS_SCALE_RATE(id, sym, cat, dens, cond, color)                         \
  if (r.materialConstant.has(BuiltInMaterial::sym)) {                          \
    auto c = r.materialConstant.get(BuiltInMaterial::sym);                     \
    c.prefactor *= factor;                                                     \
    r.materialConstant.set(Material(BuiltInMaterial::sym), c);                 \
  }
        BUILTIN_MATERIAL_LIST(PS_SCALE_RATE)
#undef PS_SCALE_RATE
      }
      auto scaled = g.stickingConstant.getDefault();
      scaled.prefactor *= factor;
      g.stickingConstant.setDefault(scaled);
#define PS_SCALE_STICKING(id, sym, cat, dens, cond, color)                     \
  if (g.stickingConstant.has(BuiltInMaterial::sym)) {                          \
    auto c = g.stickingConstant.get(BuiltInMaterial::sym);                     \
    c.prefactor *= factor;                                                     \
    g.stickingConstant.set(Material(BuiltInMaterial::sym), c);                 \
  }
      BUILTIN_MATERIAL_LIST(PS_SCALE_STICKING)
#undef PS_SCALE_STICKING
    }
    if (!found) {
      std::cerr << "--sticking: '" << label << "' is not a traced species of "
                << mech.name << "\n";
      return 1;
    }
  }

  // the analytic estimate on a flat surface, which sets the process time
  // The coverage solve on its own, which is what differs between a mechanism
  // supplied as data and one whose steady state was solved by hand. Every
  // traced flux is set to its unobstructed value, so the only work timed is
  // the per-point solve, called through the interface both models share.
  if (o.bench > 0) {
    const unsigned n = static_cast<unsigned>(o.bench);
    SmartPointer<ProcessModelBase<NumericType, D>> model;
    if (o.handwritten) {
      auto ref = SF6O2Etching<NumericType, D>::defaultParameters();
      for (size_t g = 0; g < mech.gas.size(); ++g) {
        const auto &gas = mech.gas[g];
        if (gas.label == "F_flux")
          ref.etchantFlux = gas.sourceFlux * mech.stickingOf(int(g));
        else if (gas.label == "O_flux")
          ref.passivationFlux = gas.sourceFlux * mech.stickingOf(int(g));
        else if (gas.isIonChannel)
          ref.ionFlux = gas.sourceFlux;
      }
      model = SmartPointer<SF6O2Etching<NumericType, D>>::New(ref);
    } else {
      model = SmartPointer<SurfaceChemistry<NumericType, D>>::New(mech);
    }
    auto surfaceModel = model->getSurfaceModel();
    surfaceModel->initializeCoverages(n);
    surfaceModel->initializeSurfaceData(n);

    // every label either model reads, at the flux an open field receives
    auto fluxes = SmartPointer<viennals::PointData<NumericType>>::New();
    std::vector<std::string> labels{
        "etchantFlux", "passivationFlux", "ionEnhancedFlux", "ionSputterFlux",
        "ionEnhancedPassivationFlux"};
    for (const auto &g : mech.gas)
      if (g.traced && !g.label.empty())
        labels.push_back(g.label);
    for (const auto &y : mech.ionYields)
      labels.push_back(y.label);
    for (const auto &label : labels)
      fluxes->insertNextScalarData(std::vector<NumericType>(n, 1.), label);
    const std::vector<NumericType> materialIds(
        n, static_cast<NumericType>(Material(BuiltInMaterial::Si).legacyId()));

    surfaceModel->updateCoverages(fluxes, materialIds); // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < o.benchRepeats; ++rep)
      surfaceModel->updateCoverages(fluxes, materialIds);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds =
        std::chrono::duration<double>(t1 - t0).count() / o.benchRepeats;
    std::cout << "\ncoverage solve over " << n << " points, "
              << o.benchRepeats << " repeats\n"
              << "  model      : "
              << (o.handwritten ? "hand-written closed form"
                                : "reaction file, damped Newton")
              << "\n  per call   : " << seconds * 1e3 << " ms"
              << "\n  per point  : " << seconds / n * 1e9 << " ns\n";
    return 0;
  }

  const auto gamma = mech.sourceFluxes();
  // The point solve is reported on the material the run is cut into, so a
  // mechanism whose steps are confined to one material is solved on it rather
  // than on a default that those steps do not name.
  const auto pointMaterial = MaterialMap::fromString(o.substrate);
  const auto pointConstants = mech.rateConstantsFor(pointMaterial);
  std::vector<NumericType> theta(mech.coverageNames.size(), 0.);
  mech.solveCoverages(gamma, pointConstants, theta);
  const NumericType rate =
      mech.growthRate(gamma, pointConstants, theta, pointMaterial);
  report(mech, theta, rate);

  if (rate == 0.) {
    std::cout << "the mechanism moves the surface nowhere; nothing to "
                 "simulate\n";
    return 0;
  }
  const bool etching = rate < 0.;
  // A run either removes a stated thickness of blanket material or runs for a
  // stated time. Comparing profiles across gas compositions needs the second,
  // since the blanket rate is itself one of the things that changes.
  const NumericType processTime =
      o.time > 0. ? o.time : o.thickness / std::abs(rate);

  // The stack is taller than the default domain, so it sets its own height.
  const NumericType stackTop =
      2 * o.stack * o.layer + o.oxide + o.mask + o.depth;
  const NumericType reach = o.time > 0. ? o.time * std::abs(rate) : o.thickness;
  // The substrate the feature is cut into. A mechanism's per-material blocks
  // are resolved against it, so an oxide etch has to be given an oxide.
  const auto substrateMaterial = MaterialMap::fromString(o.substrate);
  // In two dimensions the second extent is the vertical one and has to hold
  // the whole feature. In three the vertical axis is unbounded and grows on
  // its own, so the second extent is the other lateral one.
  auto domain = Domain<NumericType, D>::New(
      o.gridDelta, o.extent,
      D == 3     ? o.extent
      : o.height > 0. ? o.height
      : o.stack > 0   ? 2 * (stackTop + 20.)
                      : std::max<NumericType>(
                            200., 2 * (o.mask + o.depth + 1.4 * reach + 20.)));
  if (o.stack > 0) {
    // Alternating SiGe and Si above a silicon substrate under an oxide cap,
    // with two tapered trenches cut through every layer. The sidewall of the
    // pillar between them exposes each layer in turn, which is what a
    // per-material chemistry acts on. The layer order and the oxide cap follow
    // the SiGe stack of the ViennaPS examples.
    GeometryFactory<NumericType, D> factory(domain->getSetup());
    domain->insertNextLevelSetAsMaterial(factory.makeSubstrate(0.),
                                         Material::Si);
    NumericType top = 0.;
    for (int i = 0; i < 2 * o.stack; ++i) {
      top += o.layer;
      domain->insertNextLevelSetAsMaterial(
          factory.makeSubstrate(top),
          i % 2 == 0 ? Material::SiGe : Material::Si);
    }
    if (o.oxide > 0.) {
      top += o.oxide;
      domain->insertNextLevelSetAsMaterial(factory.makeSubstrate(top),
                                           Material::SiO2);
    }
    if (o.mask > 0.)
      domain->insertNextLevelSetAsMaterial(factory.makeMask(top, o.mask),
                                           Material::Mask);
    // The stencil is built from its base upward, so the trench is cut from the
    // over-etch depth to above the mask and its declared width is the width at
    // the bottom, the taper opening it out toward the top.
    const NumericType cut = top + o.mask;
    for (const NumericType centre : {-o.width, o.width}) {
      std::array<NumericType, D> position = {0.};
      position[0] = centre;
      position[D - 1] = -o.depth;
      auto cutout = factory.makeBoxStencil(
          position, o.width, cut + o.depth + o.gridDelta, -o.taper);
      domain->applyBooleanOperation(
          cutout, viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT);
    }
  } else if (o.fin)
    MakeFin<NumericType, D>(domain, o.width, o.depth, 0., o.mask, 0., false,
                            substrateMaterial, Material::Mask)
        .apply();
  else if constexpr (D == 2)
    // The fifth argument is the mask taper, the angle its sidewall makes with
    // the vertical, which is what a mask geometry study varies.
    MakeTrench<NumericType, D>(domain, o.width, o.depth, 0., o.mask, o.taper,
                               false, substrateMaterial, Material::Mask)
        .apply();
  else
    MakeHole<NumericType, D>(domain, o.width / 2., o.depth, 0., o.mask,
                             o.taper, HoleShape::QUARTER, substrateMaterial,
                             Material::Mask)
        .apply();
  // A deposition grows a new level set and an etch removes the substrate
  // itself, except that a mechanism which both deposits and etches needs the
  // film present from the start for its per-material rates to mean anything.
  if (!etching || !o.film.empty())
    domain->duplicateTopLevelSet(MaterialMap::fromString(
        !o.film.empty() ? o.film
        : mech.solids.empty() ? "PolySi"
                              : mech.solids.front().name));

  const std::string stem =
      o.out.empty() ? mech.name + "_" + std::to_string(D) + "D" : o.out;
  domain->saveSurfaceMesh(stem + "_initial.vtp");
  domain->saveVolumeMesh(stem + "_initial");

  if (o.time > 0.)
    std::cout << "\nprocess time = " << processTime << " s, which removes ~"
              << reach << " nm of blanket material\n";
  else
    std::cout << "\nprocess time = " << processTime << " s for ~"
              << o.thickness << " nm "
              << (etching ? "removed" : "of film") << "\n";

  SmartPointer<ProcessModelBase<NumericType, D>> model;
  if (o.handwritten) {
    // The hand-written SF6/O2 class of ViennaPS, given the same three fluxes
    // and the same ion distribution the reaction file declares, so that the
    // only difference between the two runs is where the chemistry came from.
    // The class folds the sticking of a neutral into its flux parameter: its
    // site balance uses that parameter directly, where the reaction file
    // writes the sticking and the incident flux separately and multiplies
    // them. The two describe the same surface once the flux is converted.
    auto ref = SF6O2Etching<NumericType, D>::defaultParameters();
    for (size_t g = 0; g < mech.gas.size(); ++g) {
      const auto &gas = mech.gas[g];
      if (gas.label == "F_flux")
        ref.etchantFlux = gas.sourceFlux * mech.stickingOf(int(g));
      else if (gas.label == "O_flux")
        ref.passivationFlux = gas.sourceFlux * mech.stickingOf(int(g));
      else if (gas.isIonChannel)
        ref.ionFlux = gas.sourceFlux;
    }
    ref.Ions.meanEnergy = mech.ionSource.meanEnergy;
    ref.Ions.sigmaEnergy = mech.ionSource.sigmaEnergy;
    ref.Ions.exponent = mech.ionSource.exponent;
    std::cout << "using the hand-written SF6O2Etching class: ionFlux = "
              << ref.ionFlux << ", etchantFlux = " << ref.etchantFlux
              << ", passivationFlux = " << ref.passivationFlux << "\n";
    model = SmartPointer<SF6O2Etching<NumericType, D>>::New(ref);
  } else {
    model = SmartPointer<SurfaceChemistry<NumericType, D>>::New(mech);
  }
  // With --profile the model's surface model is replaced by one that forwards
  // to it and times the coverage solve, so the solve can be reported as a
  // fraction of the run rather than in isolation.
  // The device path builds its own model through getGPUModel(), which carries
  // its own surface model, and only the transport moves to the device, so the
  // coverage solve is the same host-side class in both cases. Resolving that
  // model here rather than leaving it to the process lets the same wrapper be
  // placed on it, and a model that is already a device model is used as it
  // stands.
  SmartPointer<TimedSurfaceModel<NumericType, D>> timed;
  if (o.profile) {
#ifdef VIENNACORE_COMPILE_GPU
    if (o.gpu) {
      // The device model is normally resolved inside the process, which
      // creates the device context on the way. Resolving it here means
      // creating that context here as well.
      if (!DeviceContext::getContextFromRegistry(0))
        DeviceContext::createContext(VIENNACORE_KERNELS_PATH, 0);
      if (auto gpuModel = model->getGPUModel())
        model = gpuModel;
      else {
        std::cerr << "no device implementation of this model\n";
        return 1;
      }
    }
#endif
    timed = SmartPointer<TimedSurfaceModel<NumericType, D>>::New(
        model->getSurfaceModel());
    model->setSurfaceModel(timed);
  }

  const int snapshots = std::max(1, o.snapshots);
  Process<NumericType, D> process(domain, model, processTime / snapshots);
  if (o.intermediate) {
    Logger::setLogLevel(LogLevel::INTERMEDIATE);
    process.setIntermediateOutputPath(stem + "_");
  }
  // The device engine is picked by dimension: lines in 2D, triangles in 3D.
  process.setFluxEngineType(
      o.gpu ? (D == 3 ? FluxEngineType::GPU_TRIANGLE : FluxEngineType::GPU_LINE)
            : FluxEngineType::CPU_DISK);
  CoverageParameters coverage;
  coverage.tolerance = 1e-4;
  coverage.maxIterations = 20; // the delta metric floors on Monte-Carlo noise
  process.setParameters(coverage);
  RayTracingParameters tracing;
  tracing.raysPerPoint = o.rays;
  process.setParameters(tracing);
  // The process is applied in equal intervals so that the surface can be
  // written between them. The coverages are re-converged at the start of each
  // interval, which is what the solver does at every advection step anyway.
  for (int i = 0; i < snapshots; ++i) {
    process.apply();
    if (i + 1 < snapshots)
      domain->saveSurfaceMesh(stem + "_step" + std::to_string(i + 1) + ".vtp",
                              true);
  }

  if (o.profile && timed) {
    const double total = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - wallStart)
                             .count();
    std::cout << "\ncoverage solve inside the run\n"
              << "  calls      : " << timed->calls << "\n"
              << "  in solve   : " << timed->seconds << " s\n"
              << "  total run  : " << total << " s\n"
              << "  fraction   : " << 100. * timed->seconds / total << " %\n";
  }

  domain->saveSurfaceMesh(stem + "_final.vtp", true);
  domain->saveVolumeMesh(stem + "_final");
  std::cout << "wrote " << stem << "_initial and _final\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Logger::setLogLevel(LogLevel::INFO);
  const auto options = parse(argc, argv);
  return options.dim == 3 ? run<3>(options) : run<2>(options);
}
