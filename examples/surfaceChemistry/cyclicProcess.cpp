// A cyclic process, in which two half-cycle chemistries hand their coverages
// to one another across an advection.
//
// A cycle is a list of phases, and the half-cycles use different chemistries,
// so a process is written as two reaction files and a phase list:
//
//     dose        reactions/al2o3_tma.mechanism.json     precursor flowing
//     purge       the same chemistry, nothing flowing
//     coreactant  reactions/al2o3_h2o.mechanism.json     co-reactant flowing
//     purge       the same chemistry, nothing flowing
//
// The coverages carry from each step into the next, so what the dose leaves on
// the surface is what the co-reactant acts on, which is the whole content of a
// cycle and is why the coverages here are integrated in time rather than
// solved at steady state. A dose has no steady state: it saturates.
//
//     ./cyclicProcess --cycles 20 --width 60 --depth 400
//
// Any pair of half-cycle reaction files runs through it, since each pulse
// flows exactly the species its own mechanism traces:
//
//     ./cyclicProcess --dose-file reactions/sin_peald_dis_dose.mechanism.json \
//                     --coreactant-file reactions/sin_peald_n2h2_plasma.mechanism.json \
//                     --film SiN --cycles 50
//
// --lateral replaces the trench with a lateral high-aspect-ratio cavity, the
// structure conformality is measured in, and switches the driver to
// micrometres. That run also writes the film thickness along the cavity.
//
// The flux engine defaults to the GPU when one is available; --engine cpu
// forces the host path.
//
// Growing into a deep feature is the point of running this on a geometry at
// all: the conformality of a half-cycle is set by the sticking of its
// adsorption steps, which the reaction files carry.

#include <geometries/psMakeTrench.hpp>
#include <models/psChemicalMechanismIO.hpp>
#include <models/psSurfaceChemistry.hpp>
#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psPlanarize.hpp>
#include <psToDiskMesh.hpp>

#include "../atomicLayerDeposition/geometry.hpp"

#include <fstream>
#include <iostream>
#include <string>

using namespace viennaps;

namespace {

using NumericType = double;

struct Options {
  std::string dose = "reactions/al2o3_tma.mechanism.json";
  std::string coreactant = "reactions/al2o3_h2o.mechanism.json";
  std::string name = "atomicLayerProcess";
  std::string film = "Al2O3"; // the material label given to the grown film
  int cycles = 25;
  NumericType doseTime = 0.2;   // s
  NumericType purgeTime = 3.0;
  NumericType coreactantTime = 15.0;
  NumericType gridDelta = 2.;
  NumericType width = 60.;
  NumericType depth = 300.;
  int rays = 1000;
  // Coverage sub-steps per pulse. Every one of them re-traces the fluxes,
  // because the sticking depends on the coverage, so this is what a cycle
  // costs. Refine it until the growth per cycle stops moving.
  int doseSteps = 20;
  int coreactantSteps = 30;
  // A lateral high-aspect-ratio cavity instead of a trench. The dimensions
  // are those of the PillarHall test structure, in micrometres, and the
  // conformality is read out along the gap rather than down a sidewall.
  bool lateral = false;
  NumericType gapLength = 100.;
  NumericType gapHeight = 0.5;
  NumericType openingWidth = 10.;
  NumericType openingDepth = 0.5;
  NumericType xPad = 0.5;
  std::string out = "cyclicProcess";
  // Write a mesh per coverage sub-step, carrying the coverages, so the
  // surface can be watched saturating through a pulse and down the trench.
  // One file per sub-step per cycle, so keep --cycles small when using it.
  bool intermediate = false;
  std::string engine = "auto";  // auto | cpu | gpu
  double maxChange = 1e-3;  // transient integrator accuracy; ~5% at 1e-3
  // The termination the substrate carries before the first pulse. An atomic
  // layer process starts on a terminated surface, and a mechanism whose first
  // step consumes that termination deposits nothing without it.
  std::vector<std::pair<std::string, double>> initial;
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
    else if (arg == "--coreactant")
      o.coreactantTime = std::stod(next());
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
    else if (arg == "--coreactant-steps")
      o.coreactantSteps = std::stoi(next());
    else if (arg == "--out")
      o.out = next();
    else if (arg == "--intermediate")
      o.intermediate = true;
    else if (arg == "--dose-file")
      o.dose = next();
    else if (arg == "--coreactant-file")
      o.coreactant = next();
    else if (arg == "--name")
      o.name = next();
    else if (arg == "--film")
      o.film = next();
    else if (arg == "--engine")
      o.engine = next();
    else if (arg == "--lateral")
      o.lateral = true;
    else if (arg == "--gap-length")
      o.gapLength = std::stod(next());
    else if (arg == "--gap-height")
      o.gapHeight = std::stod(next());
    else if (arg == "--initial") {
      const std::string spec = next();
      const auto eq = spec.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--initial wants SPECIES=fraction, got " << spec << "\n";
        std::exit(1);
      }
      o.initial.emplace_back(spec.substr(0, eq), std::stod(spec.substr(eq + 1)));
    }
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

  // The trench is quoted in nanometres and the lateral cavity in
  // micrometres, and every rate constant is converted to the active unit.
  units::Length::setUnit(o.lateral ? "um" : "nm");
  units::Time::setUnit("s");
  const NumericType perAngstrom = o.lateral ? 1.e4 : 10.;
  if (o.intermediate)
    Logger::setLogLevel(LogLevel::INTERMEDIATE);

  auto dose = readChemicalMechanism<NumericType>(o.dose);
  auto coreactant = readChemicalMechanism<NumericType>(o.coreactant);

  auto domain = Domain<NumericType, D>::New();
  if (o.lateral) {
    makeT<NumericType, D>(domain, o.gridDelta, o.openingDepth, o.openingWidth,
                          o.gapLength, o.gapHeight, o.xPad, Material::Si);
  } else {
    domain = Domain<NumericType, D>::New(o.gridDelta, 200., 400.);
    MakeTrench<NumericType, D>(domain, o.width, o.depth, 0., 0., 0., false,
                               Material::Si, Material::Mask)
        .apply();
  }
  domain->duplicateTopLevelSet(MaterialMap::fromString(o.film));
  domain->saveSurfaceMesh(o.out + "_initial.vtp");

  auto model = SmartPointer<SurfaceChemistry<NumericType, D>>::New();
  model->addMechanism("dose", dose);
  model->addMechanism("coreactant", coreactant);
  model->setAtomicLayerProcess();
  model->setMaxCoverageChange(o.maxChange);
  for (const auto &[name, value] : o.initial)
    model->setInitialCoverage(name, static_cast<NumericType>(value));
  model->setProcessName(o.name);

  // The species each pulse flows. A purge names none, so only the thermal
  // steps of that half-cycle's chemistry run through it.
  AtomicLayerProcessParameters alp;
  alp.numCycles = o.cycles;
  // A pulse flows exactly the species its own mechanism traces, so the driver
  // names no species and works for any pair of half-cycle reaction files.
  auto tracedOf = [](const ChemicalMechanism<NumericType> &m) {
    std::vector<std::string> labels;
    for (const auto &g : m.gas)
      if (g.traced && !g.label.empty() && !g.isIonChannel)
        labels.push_back(g.label);
    return labels;
  };
  alp.addPhase("dose", o.doseTime, o.doseTime / o.doseSteps,
               tracedOf(dose), "dose");
  alp.addPhase("purge_dose", o.purgeTime, o.purgeTime / 4., {}, "dose");
  alp.addPhase("coreactant", o.coreactantTime,
               o.coreactantTime / o.coreactantSteps,
               tracedOf(coreactant), "coreactant");
  alp.addPhase("purge_coreactant", o.purgeTime, o.purgeTime / 4., {},
               "coreactant");

  std::cout << o.name + ": " << o.cycles << " cycles of "
            << o.doseTime << " s dose / " << o.purgeTime << " s purge / "
            << o.coreactantTime << " s co-reactant / " << o.purgeTime
            << " s purge\n";
  if (o.lateral)
    std::cout << "lateral cavity " << o.gapLength << " um long, " << o.gapHeight
              << " um high (aspect ratio " << o.gapLength / o.gapHeight
              << ")\n\n";
  else
    std::cout << "trench " << o.width << " nm wide, " << o.depth
              << " nm deep (aspect ratio " << o.depth / o.width << ")\n\n";

  // Only the transport moves to the device; the coverage integration runs on
  // the host either way, so the two engines agree to within ray noise.
  const bool haveGPU = gpuAvailable();
  if (o.engine == "gpu" && !haveGPU) {
    std::cerr << "no GPU available: build with VIENNAPS_USE_GPU=ON, or use "
                 "--engine cpu\n";
    return 1;
  }
  const bool useGPU = o.engine == "gpu" || (o.engine == "auto" && haveGPU);
  std::cout << "flux engine: " << (useGPU ? "GPU" : "CPU")
            << (o.engine == "auto" ? " (auto)" : "") << "\n";

  Process<NumericType, D> process(domain, model);
  process.setFluxEngineType(useGPU ? FluxEngineType::GPU_LINE
                                   : FluxEngineType::CPU_DISK);
  process.setParameters(alp);
  RayTracingParameters tracing;
  tracing.raysPerPoint = o.rays;
  process.setParameters(tracing);
  process.apply();

  std::cout << "\ngrowth per cycle on the open field = "
            << model->growthPerCycle() * perAngstrom << " A/cycle\n";

  // Conformality along a lateral cavity is the film thickness as a function of
  // penetration depth, which is what the structure is built to measure. The
  // domain is planarized at half the gap height so that the cut crosses the
  // film grown on the cavity floor, and the surface height along that cut is
  // the deposited thickness.
  if (o.lateral) {
    auto measured = SmartPointer<Domain<NumericType, D>>::New();
    measured->deepCopy(domain);
    Planarize<NumericType, D>(measured, o.gapHeight / 2.).apply();
    auto mesh = viennals::Mesh<NumericType>::New();
    ToDiskMesh<NumericType, D>(measured, mesh).apply();
    std::ofstream file(o.out + "_profile.txt");
    file << "# position_um height_um\n";
    for (const auto &node : mesh->nodes)
      file << node[0] << " " << node[1] << "\n";
    file.close();
    std::cout << "wrote " << o.out << "_profile.txt ("
              << mesh->nodes.size() << " points)\n";
  }

  domain->saveSurfaceMesh(o.out + "_final.vtp", true);
  domain->saveVolumeMesh(o.out + "_final");
  std::cout << "\nwrote " << o.out << "_initial.vtp, " << o.out
            << "_final.vtp and " << o.out
            << "_final_volume.vtu\n"
               "  open the .vtu in ParaView and colour by 'Material' to see "
               "the film against the substrate\n";
  return 0;
}
