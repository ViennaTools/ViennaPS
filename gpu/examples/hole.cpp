#include <geometries/psMakeHole.hpp>
#include <models/psSF6O2Etching.hpp>

#include <curtTrace.hpp>

#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMeshRefined.hpp>

#include <utElementToPointData.hpp>

using namespace viennaps;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;

  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;
  CreateContext(context);

  const NumericType gridDelta = .9;
  const NumericType extent = 60.;
  const NumericType holeRadius = 15.;
  const NumericType maskHeight = 100.;

  const NumericType time = 12.;
  const NumericType ionFlux = 10.;
  const NumericType etchantFlux = 2000.;
  const NumericType oxygenFlux = 1000.;
  const NumericType ionEnergy = 100.;
  const NumericType ionSigma = 10.;
  const NumericType exponent = 1000.;

  auto domain = SmartPointer<Domain<NumericType, D>>::New();
  MakeHole<NumericType, D>(domain, gridDelta, extent, extent, holeRadius,
                           maskHeight, 0.f, 0.f, false, true, Material::Si)
      .apply();

  domain->saveSurfaceMesh("hole_initial.vtp");

  auto diskMesh = SmartPointer<viennals::Mesh<NumericType>>::New();
  viennals::ToDiskMesh<NumericType, D> diskMesher(diskMesh);
  for (const auto ls : domain->getLevelSets()) {
    diskMesher.insertNextLevelSet(ls);
  }

  auto elementKdTree = SmartPointer<KDTree<float, Vec3Df>>::New();
  auto surfMesh = SmartPointer<viennals::Mesh<float>>::New();
  viennals::ToSurfaceMeshRefined<NumericType, float, D> surfMesher(
      domain->getLevelSets().back(), surfMesh, elementKdTree);

  SF6O2Parameters<NumericType> params;

  gpu::Particle<NumericType> ion;
  ion.name = "ion";
  ion.dataLabels.push_back("ionSputteringRate");
  ion.dataLabels.push_back("ionEnhancedRate");
  ion.dataLabels.push_back("oxygenSputteringRate");
  ion.sticking = 1.f;
  ion.cosineExponent = params.Ions.exponent;

  gpu::Particle<NumericType> etchant;
  etchant.name = "etchant";
  etchant.dataLabels.push_back("etchantRate");
  etchant.sticking = params.beta_F;
  etchant.cosineExponent = 1.f;

  gpu::Particle<NumericType> oxygen;
  oxygen.name = "oxygen";
  oxygen.dataLabels.push_back("oxygenRate");
  oxygen.sticking = params.beta_O;
  oxygen.cosineExponent = 1.f;

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setNumberOfRaysPerPoint(5000);
  tracer.setUseRandomSeeds(false);
  tracer.insertNextParticle(ion);
  tracer.insertNextParticle(etchant);
  tracer.insertNextParticle(oxygen);

  Timer timer;
  timer.start();
  diskMesher.apply();
  surfMesher.apply();
  gpu::TriangleMesh mesh;
  mesh.gridDelta = gridDelta;
  mesh.vertices = surfMesh->nodes;
  mesh.triangles = surfMesh->triangles;
  mesh.minimumExtent = surfMesh->minimumExtent;
  mesh.maximumExtent = surfMesh->maximumExtent;
  timer.finish();
  std::cout << "Meshing time: " << timer.currentDuration * 1e-6 << " ms"
            << std::endl;

  tracer.setGeometry(mesh);
  tracer.setPipeline("SF6O2Pipeline");
  tracer.setParameters(params);
  CudaBuffer cellDataBuffer;
  cellDataBuffer.allocInit(mesh.triangles.size() * 2, 0.f);
  tracer.setElementData(cellDataBuffer, 2);

  tracer.prepareParticlePrograms();

  tracer.apply();

  auto elementData = SmartPointer<viennals::PointData<NumericType>>::New();
  tracer.downloadResultsToPointData(surfMesh->getCellData());

  auto pointData = SmartPointer<viennals::PointData<NumericType>>::New();
  gpu::ElementToPointData<NumericType>(tracer.getResults(), pointData,
                                       tracer.getParticles(), elementKdTree,
                                       diskMesh, surfMesh, gridDelta)
      .apply();
  diskMesh->cellData = *pointData;
  viennals::VTKWriter<NumericType>(diskMesh, "testDisk.vtp").apply();

  // std::vector<NumericType> flux(tracer.getNumberOfElements());
  // tracer.getFlux(flux.data(), 0, 0);
  // std::cout << "flux " << flux.back() << std::endl;
  // surfMesh->getCellData().insertNextScalarData(flux, "Flux");
  viennals::VTKWriter<NumericType>(surfMesh, "test.vtp").apply();

  // auto surfModel =
  //     SmartPointer<SF6O2Implementation::SurfaceModel<NumericType, D>>::New(
  //         ionFlux, etchantFlux, oxygenFlux, -100000);
  // auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);
  // auto model = SmartPointer<gpu::ProcessModel<NumericType>>::New();

  // model->insertNextParticleType(ion);
  // model->insertNextParticleType(etchant);
  // model->insertNextParticleType(oxygen);
  // model->setSurfaceModel(surfModel);
  // model->setVelocityField(velField);
  // model->setProcessName("SF6O2Etching");
  // model->setPipelineFileName("SF6O2Pipeline");

  // gpu::Process<NumericType, D> process(context);
  // process.setDomain(domain);
  // process.setProcessModel(model);
  // process.setNumberOfRaysPerPoint(1000);
  // process.setMaxCoverageInitIterations(10);
  // process.setProcessDuration(time);
  // process.setSmoothFlux(1.);
  // process.apply();

  // domain->saveSurfaceMesh("hole_etched_t12.vtp");
}