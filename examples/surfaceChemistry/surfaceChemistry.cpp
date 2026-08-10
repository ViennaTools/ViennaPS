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

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <models/psChemicalMechanismIO.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>


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
  if (mech.ionSource.present)
    std::cout << "  [ion, " << mech.ionSource.meanEnergy << " eV]";
  std::cout << "\nreactions   :\n";
  for (const auto &r : mech.reactions)
    std::cout << "   " << r.equation << "\n";
  std::cout << "steady state:\n";
  for (size_t i = 0; i < theta.size(); ++i)
    std::cout << "   theta_" << mech.coverageNames[i] << " = " << std::scientific
              << std::setprecision(6) << theta[i] << "\n";
  std::cout << "   " << (rate < 0. ? "etch rate  " : "growth rate")
            << " = " << rate << " nm/s\n"
            << std::defaultfloat;
}

template <int D> int run(const Options &o) {
  units::Length::setUnit("nm");
  units::Time::setUnit("s");

  auto mech = readChemicalMechanism<NumericType>(o.reactions);

  // the analytic estimate on a flat surface, which sets the process time
  const auto gamma = mech.sourceFluxes();
  std::vector<NumericType> theta(mech.coverageNames.size(), 0.);
  mech.solveCoverages(gamma, mech.rateConstants(), theta);
  const NumericType rate = mech.growthRate(gamma, mech.rateConstants(), theta);
  report(mech, theta, rate);

  if (rate == 0.) {
    std::cout << "the mechanism moves the surface nowhere; nothing to "
                 "simulate\n";
    return 0;
  }
  const bool etching = rate < 0.;
  const NumericType processTime = o.thickness / std::abs(rate);

  auto domain = Domain<NumericType, D>::New(o.gridDelta, 200., 200.);
  if constexpr (D == 2)
    MakeTrench<NumericType, D>(domain, o.width, o.depth, 0., o.mask, 0., false,
                               Material::Si, Material::Mask)
        .apply();
  else
    MakeHole<NumericType, D>(domain, o.width / 2., o.depth, 0., o.mask, 0.,
                             HoleShape::QUARTER, Material::Si, Material::Mask)
        .apply();
  // a deposition grows a new level set; an etch removes the substrate itself
  if (!etching)
    domain->duplicateTopLevelSet(
        MaterialMap::fromString(mech.solids.empty() ? "PolySi"
                                                    : mech.solids.front().name));

  const std::string stem = mech.name + "_" + std::to_string(D) + "D";
  domain->saveSurfaceMesh(stem + "_initial.vtp");
  domain->saveVolumeMesh(stem + "_initial");

  std::cout << "\nprocess time = " << processTime << " s for ~" << o.thickness
            << " nm " << (etching ? "removed" : "of film") << "\n";

  auto model = SmartPointer<SurfaceChemistry<NumericType, D>>::New(mech);
  Process<NumericType, D> process(domain, model, processTime);
  process.setFluxEngineType(o.gpu ? FluxEngineType::GPU_LINE
                                  : FluxEngineType::CPU_DISK);
  CoverageParameters coverage;
  coverage.tolerance = 1e-4;
  coverage.maxIterations = 20; // the delta metric floors on Monte-Carlo noise
  process.setParameters(coverage);
  RayTracingParameters tracing;
  tracing.raysPerPoint = o.rays;
  process.setParameters(tracing);
  process.apply();

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
