// PE-ALD of SiNx from diiodosilane and an N2-H2 plasma, in a trench.
//
// A cycle is four steps, and two of them use a different chemistry from the
// other two, so the process is written as two reaction files and a phase list:
//
//     dose    reactions/sin_peald_dis_dose.yaml      SiH2I2 flowing
//     purge   the same chemistry, nothing flowing
//     plasma  reactions/sin_peald_n2h2_plasma.yaml   N, H, NH, N2+ flowing
//     purge   the same chemistry, nothing flowing
//
// The coverages carry from each step into the next -- what the dose leaves on
// the surface is what the plasma acts on -- which is the whole content of an
// ALD cycle and is why the coverages here are integrated in time rather than
// solved at steady state. A dose has no steady state: it saturates.
//
//     ./sinPEALD --cycles 50 --width 60 --depth 400
//
// Growing into a deep trench is the point of running this on a geometry at
// all: the plasma step's conformality is set by the recombination probability
// of the N and H radicals, which the two reaction files carry as the sticking
// of their last adsorption step.
//
//     M. Zeghouane et al., Mater. Sci. Semicond. Process. 184 (2024) 108851.

#include <geometries/psMakeTrench.hpp>
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>

#include <iostream>
#include <string>

using namespace viennaps;

namespace {

using NumericType = double;

struct Options {
  std::string dose = "reactions/sin_peald_dis_dose.mechanism.json";
  std::string plasma = "reactions/sin_peald_n2h2_plasma.mechanism.json";
  int cycles = 25;
  NumericType doseTime = 0.2;   // s, the paper's standard process
  NumericType purgeTime = 3.0;
  NumericType plasmaTime = 15.0;
  NumericType gridDelta = 2.;
  NumericType width = 60.;
  NumericType depth = 300.;
  int rays = 1000;
  // Coverage sub-steps per pulse. Every one of them re-traces the fluxes,
  // because the sticking depends on the coverage, so this is what a cycle
  // costs. Refine it until the growth per cycle stops moving.
  int doseSteps = 20;
  int plasmaSteps = 30;
  std::string out = "sinPEALD";
  // Write a mesh per coverage sub-step, carrying the coverages, so the
  // surface can be watched saturating through a pulse and down the trench.
  // One file per sub-step per cycle, so keep --cycles small when using it.
  bool intermediate = false;
  bool gpu = false;
  double maxChange = 1e-3;  // transient integrator accuracy; ~5% at 1e-3
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
    if (arg == "--cycles")
      o.cycles = std::stoi(next());
    else if (arg == "--dose")
      o.doseTime = std::stod(next());
    else if (arg == "--plasma")
      o.plasmaTime = std::stod(next());
    else if (arg == "--purge")
      o.purgeTime = std::stod(next());
    else if (arg == "--width")
      o.width = std::stod(next());
    else if (arg == "--depth")
      o.depth = std::stod(next());
    else if (arg == "--grid")
      o.gridDelta = std::stod(next());
    else if (arg == "--rays")
      o.rays = std::stoi(next());
    else if (arg == "--dose-steps")
      o.doseSteps = std::stoi(next());
    else if (arg == "--plasma-steps")
      o.plasmaSteps = std::stoi(next());
    else if (arg == "--out")
      o.out = next();
    else if (arg == "--intermediate")
      o.intermediate = true;
    else if (arg == "--gpu")
      o.gpu = true;
    else if (arg == "--max-change")
      o.maxChange = std::stod(next());
    else {
      std::cerr << "unknown option " << arg << "\n";
      std::exit(1);
    }
  }
  return o;
}

} // namespace

int main(int argc, char **argv) {
  constexpr int D = 2;
  const auto o = parse(argc, argv);

  units::Length::setUnit("nm");
  units::Time::setUnit("s");
  if (o.intermediate)
    Logger::setLogLevel(LogLevel::INTERMEDIATE);

  auto dose = readChemicalMechanism<NumericType>(o.dose);
  auto plasma = readChemicalMechanism<NumericType>(o.plasma);

  auto domain = Domain<NumericType, D>::New(o.gridDelta, 200., 400.);
  MakeTrench<NumericType, D>(domain, o.width, o.depth, 0., 0., 0., false,
                             Material::Si, Material::Mask)
      .apply();
  domain->duplicateTopLevelSet(Material::SiN);
  domain->saveSurfaceMesh(o.out + "_initial.vtp");

  auto model = SmartPointer<SurfaceChemistry<NumericType, D>>::New();
  model->addMechanism("dose", dose);
  model->addMechanism("plasma", plasma);
  model->setAtomicLayerProcess();
  model->setMaxCoverageChange(o.maxChange);
  model->setProcessName("sinPEALD");

  // The species each pulse flows. A purge names none, so only the thermal
  // steps of that half-cycle's chemistry run through it.
  AtomicLayerProcessParameters alp;
  alp.numCycles = o.cycles;
  alp.addPhase("dose", o.doseTime, o.doseTime / o.doseSteps,
               {"SiH2I2_flux"}, "dose");
  alp.addPhase("purge_dose", o.purgeTime, o.purgeTime / 4., {}, "dose");
  alp.addPhase("plasma", o.plasmaTime, o.plasmaTime / o.plasmaSteps,
               {"N_flux", "H_flux", "NH_flux"}, "plasma");
  alp.addPhase("purge_plasma", o.purgeTime, o.purgeTime / 4., {}, "plasma");

  std::cout << "SiNx PE-ALD: " << o.cycles << " cycles of "
            << o.doseTime << " s dose / " << o.purgeTime << " s purge / "
            << o.plasmaTime << " s plasma / " << o.purgeTime << " s purge\n"
            << "trench " << o.width << " nm wide, " << o.depth
            << " nm deep (aspect ratio " << o.depth / o.width << ")\n\n";

  Process<NumericType, D> process(domain, model);
  process.setFluxEngineType(o.gpu ? FluxEngineType::GPU_LINE
                                  : FluxEngineType::CPU_DISK);
  process.setParameters(alp);
  RayTracingParameters tracing;
  tracing.raysPerPoint = o.rays;
  process.setParameters(tracing);
  process.apply();

  std::cout << "\ngrowth per cycle on the open field = "
            << model->growthPerCycle() * 10. << " A/cycle"
            << "   (measured 0.36-0.40 A/cycle at 300 C)\n";

  domain->saveSurfaceMesh(o.out + "_final.vtp", true);
  domain->saveVolumeMesh(o.out + "_final");
  std::cout << "\nwrote " << o.out << "_initial.vtp, " << o.out
            << "_final.vtp and " << o.out
            << "_final_volume.vtu\n"
               "  open the .vtu in ParaView and colour by 'Material' to see "
               "the film against the substrate\n";
  return 0;
}
