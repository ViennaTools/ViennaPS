#include <viennaps.hpp>

using namespace viennaps;

int main(int argc, char *argv[]) {

  using NumericType = double;
  constexpr int D = 3;

  // Parse the parameters
  util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    // Try default config file
    params.readConfigFile("config.txt");
    if (params.m.empty()) {
      std::cout << "No configuration file provided!" << std::endl;
      std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
      return 1;
    }
  }

  auto geometry = Domain<NumericType, D>::New(
      params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));
  MakeHole<NumericType, D>(geometry, params.get("trenchWidth"),
                           params.get("trenchHeight"), params.get("taperAngle"))
      .apply();

  auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
      params.get("rate"), params.get("stickingProbability"),
      params.get("sourcePower"));

  Process<NumericType, D> process(geometry, model);
  auto flux = process.calculateFlux();

  PointCloud<NumericType> cloud;
  cloud.positions = flux->getNodes();
  cloud.normals = *flux->getCellData().getVectorData("Normals");

  using GraphSolver = SurfaceDiffusionSolver<NumericType>;
  using GraphStencil = SurfaceDiffusionStencil<NumericType>;

  GraphStencil::Parameters stencilParams;
  stencilParams.kNeighbors = 16;
  GraphSolver solver(GraphStencil(cloud, stencilParams));

  const double dt = 1e-3;
  auto currentFlux = *flux->getCellData().getScalarData("particleFlux");
  for (int step = 0; step < 10000; ++step) {
    currentFlux = solver.stepExplicit(currentFlux, dt, 1.);

    if (step % 100 == 0) {
      flux->getCellData().insertReplaceScalarData(currentFlux, "particleFlux");
      viennals::VTKWriter<NumericType>(flux, "surface_diffusion_step_" +
                                                 std::to_string(step))
          .apply();
    }
  }

  return 0;
}
