#include <geometries/psMakePlane.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psIsotropicProcess.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

#include <models/psgFaradayCageEtching.hpp>
#include <psgProcess.hpp>

namespace ps = viennaps;
namespace ls = viennals;

template <class NumericType>
void FluxOnMesh(
    ls::PointData<NumericType> &pointData,
    ps::KDTree<NumericType, ps::Vec3D<NumericType>> const &pointKdTree,
    const ps::SmartPointer<ls::Mesh<float>> &surfaceMesh) {

  auto numData = pointData.getScalarDataSize();
  const auto &elements = surfaceMesh->triangles;
  const auto &nodes = surfaceMesh->nodes;
  auto numElements = elements.size();
  std::vector<float> elementData(numData * numElements);
  std::vector<unsigned> dataIdx(numData);
  auto material = pointData.getScalarData("MaterialIds");

  std::vector<float> data(numElements);
  for (unsigned i = 0; i < numData; i++) {
    auto label = pointData.getScalarDataLabel(i);
    surfaceMesh->getCellData().insertReplaceScalarData(data, label);
    dataIdx[i] = surfaceMesh->getCellData().getScalarDataIndex(label);
  }

#pragma omp parallel for
  for (unsigned i = 0; i < numElements; i++) {
    auto &elIdx = elements[i];
    std::array<NumericType, 3> elementCenter{
        static_cast<NumericType>(
            (nodes[elIdx[0]][0] + nodes[elIdx[1]][0] + nodes[elIdx[2]][0]) /
            3.f),
        static_cast<NumericType>(
            (nodes[elIdx[0]][1] + nodes[elIdx[1]][1] + nodes[elIdx[2]][1]) /
            3.f),
        static_cast<NumericType>(
            (nodes[elIdx[0]][2] + nodes[elIdx[1]][2] + nodes[elIdx[2]][2]) /
            3.f)};

    auto closestPoint = pointKdTree.findNearest(elementCenter);

    for (unsigned j = 0; j < numData; j++) {
      auto value = pointData.getScalarData(j)->at(closestPoint->first);
      if (material->at(closestPoint->first) == 0)
        value = 0;
      surfaceMesh->getCellData().getScalarData(dataIdx[j])->at(i) =
          static_cast<float>(value);
    }
  }
}

template <class NumericType>
auto createMesh(const NumericType W1, const NumericType W2,
                const NumericType W3, const NumericType W4,
                const NumericType L1, const NumericType L2,
                const NumericType bottomLength, const NumericType z,
                const bool orient) {
  auto mesh = ps::SmartPointer<ls::Mesh<NumericType>>::New();

  // bottom rectangle
  mesh->insertNextNode({-bottomLength, -W1 / 2.f, z}); // 0
  mesh->insertNextNode({0.0f, -W1 / 2.f, z});          // 1
  mesh->insertNextNode({0.0f, W1 / 2.f, z});           // 2
  mesh->insertNextNode({-bottomLength, W1 / 2.f, z});  // 3

  if (orient) {
    mesh->insertNextTriangle({0, 1, 2});
    mesh->insertNextTriangle({0, 2, 3});
  } else {
    mesh->insertNextTriangle({2, 1, 0});
    mesh->insertNextTriangle({3, 2, 0});
  }

  // tip lower part
  mesh->insertNextNode({0.0f, -W2 / 2.f, z}); // 4
  mesh->insertNextNode({L1, -W3 / 2.f, z});   // 5
  mesh->insertNextNode({L1, W3 / 2.f, z});    // 6
  mesh->insertNextNode({0.0f, W2 / 2.f, z});  // 7

  if (orient) {
    mesh->insertNextTriangle({1, 5, 6});
    mesh->insertNextTriangle({1, 6, 2});
    mesh->insertNextTriangle({1, 4, 5});
    mesh->insertNextTriangle({2, 6, 7});
  } else {
    mesh->insertNextTriangle({6, 5, 1});
    mesh->insertNextTriangle({2, 6, 1});
    mesh->insertNextTriangle({5, 4, 1});
    mesh->insertNextTriangle({7, 6, 2});
  }

  // tip upper part
  mesh->insertNextNode({L1 + L2, -W4 / 2.f, z}); // 8
  mesh->insertNextNode({L1 + L2, W4 / 2.f, z});  // 9

  if (orient) {
    mesh->insertNextTriangle({6, 8, 9});
    mesh->insertNextTriangle({6, 5, 8});
  } else {
    mesh->insertNextTriangle({9, 8, 6});
    mesh->insertNextTriangle({8, 5, 6});
  }

  return mesh;
}

template <class NumericType, int D>
void MakeGeometry(ps::SmartPointer<ps::Domain<NumericType, D>> &domain,
                  const NumericType gridDelta, const NumericType W1,
                  const NumericType W2, const NumericType W3,
                  const NumericType W4, const NumericType L1,
                  const NumericType L2, const NumericType xPad,
                  const NumericType yPad, const NumericType height) {

  const NumericType bottomLength = 5500.;

  auto upper =
      createMesh<float>(W1, W2, W3, W4, L1, L2, bottomLength, height, true);
  auto lower =
      createMesh<float>(W1, W2, W3, W4, L1, L2, bottomLength, 0., false);
  upper->append(*lower);

  upper->insertNextTriangle({11, 1, 0});
  upper->insertNextTriangle({10, 11, 0});
  upper->insertNextTriangle({10, 0, 3});
  upper->insertNextTriangle({10, 3, 13});
  upper->insertNextTriangle({13, 3, 2});
  upper->insertNextTriangle({13, 2, 12});
  upper->insertNextTriangle({5, 4, 14});
  upper->insertNextTriangle({5, 14, 15});
  upper->insertNextTriangle({8, 5, 15});
  upper->insertNextTriangle({8, 15, 18});
  upper->insertNextTriangle({9, 8, 18});
  upper->insertNextTriangle({9, 18, 19});
  upper->insertNextTriangle({6, 9, 19});
  upper->insertNextTriangle({6, 19, 16});
  upper->insertNextTriangle({7, 6, 16});
  upper->insertNextTriangle({7, 16, 17});
  upper->insertNextTriangle({4, 1, 11});
  upper->insertNextTriangle({4, 11, 14});
  upper->insertNextTriangle({2, 7, 17});
  upper->insertNextTriangle({2, 17, 12});

  double bounds[2 * D] = {-bottomLength - xPad,
                          L1 + L2 + xPad,
                          -W1 - yPad,
                          W1 + yPad,
                          -height,
                          height};
  typename ls::BoundaryConditionEnum boundaryCons[D] = {
      ls::BoundaryConditionEnum::PERIODIC_BOUNDARY,
      ls::BoundaryConditionEnum::PERIODIC_BOUNDARY,
      ls::BoundaryConditionEnum::INFINITE_BOUNDARY};
  auto levelSet = ls::SmartPointer<ls::Domain<NumericType, D>>::New(
      bounds, boundaryCons, gridDelta);
  ls::FromSurfaceMesh<NumericType, D>(levelSet, upper).apply();
  domain->insertNextLevelSetAsMaterial(levelSet, ps::Material::Mask);
}

template <class NumericType, int D>
void clean(ps::SmartPointer<ps::Domain<NumericType, D>> geometry,
           const NumericType smoothingSize) {
  // clean up process
  {
    auto model = ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(
        -1., ps::Material::Mask);
    ps::Process<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setProcessDuration(smoothingSize);
    process.setTimeStepRatio(0.1);
    process.apply();
  }

  {
    auto model = ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(
        1., ps::Material::Mask);
    ps::Process<NumericType, D> process;
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setProcessDuration(smoothingSize);
    process.setTimeStepRatio(0.1);
    process.apply();
  }
}

int main(int argc, char *argv[]) {
  using NumericType = float;
  constexpr int D = 3;
  ps::Logger::setLogLevel(ps::LogLevel::INFO);

  ps::Context context;
  context.create();
  omp_set_num_threads(16);
  std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;

  // Parse the parameters
  ps::util::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <parameter file>" << std::endl;
    return EXIT_FAILURE;
  }

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
  const auto W1 = params.get<NumericType>("W1");
  const auto W2 = params.get<NumericType>("W2");
  const auto W3 = params.get<NumericType>("W3");
  const auto W4 = params.get<NumericType>("W4");

  if (W3 <= W2) {
    std::cout << "W3 must be greater than W2" << std::endl;
    return EXIT_FAILURE;
  }
  if (W2 <= W1) {
    std::cout << "W2 must be greater than W1" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Output file: ";
  std::cout << params.get<std::string>("outputFile") << std::endl;

  MakeGeometry<NumericType, D>(
      geometry, params.get("gridDelta"), W1, W2, W3, W4, params.get("L1"),
      params.get("L2"), params.get("boundaryPadding_x"),
      params.get("boundaryPadding_y"), params.get("maskHeight"));
  geometry->saveSurfaceMesh("mask.vtp");

  if (params.get("scalingSize") > 0.) {
    auto scalingModel =
        ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(
            params.get("scalingSize") * 1.5);
    ps::Process<NumericType, D>(geometry, scalingModel, 1).apply();

    auto scalingModelRed =
        ps::SmartPointer<ps::IsotropicProcess<NumericType, D>>::New(
            -params.get("scalingSize") * 0.5);
    ps::Process<NumericType, D>(geometry, scalingModelRed, 1).apply();

    ps::Planarize<NumericType, D>(geometry, params.get("maskHeight")).apply();
    geometry->saveSurfaceMesh("scaled_mask.vtp");
  }

  ps::MakePlane<NumericType, D>(geometry, 0., ps::Material::Si, true)
      .apply(); // add Si as the base material
  geometry->saveSurfaceMesh("initial.vtp");

  {
    typename ps::DirectionalProcess<NumericType, D>::RateSet rateSet;
    rateSet.direction = {0.};
    rateSet.direction[D - 1] = -1.;
    rateSet.directionalVelocity = -1.;
    rateSet.isotropicVelocity = 0.;
    rateSet.maskMaterials = std::vector<ps::Material>{ps::Material::Mask};
    rateSet.calculateVisibility = false;

    auto etchModel =
        ps::SmartPointer<ps::DirectionalProcess<NumericType, D>>::New(rateSet);
    ps::Process<NumericType, D>(geometry, etchModel,
                                params.get("verticalDepth"))
        .apply();
  }
  geometry->saveSurfaceMesh("vertical_etch.vtp");

  NumericType time = 0.;
  NumericType duration = params.get("processTime");
  int counter = 0;
  NumericType cageAngle = 0.;
  NumericType tiltAngle = params.get("tiltAngle");
  NumericType timePerAngle = params.get("timePerAngle");
  while (time < duration) {
    // faraday cage source setup
    auto model =
        ps::SmartPointer<ps::gpu::FaradayCageEtching<NumericType, D>>::New(
            1.0, params.get("stickProbability"), params.get("sourcePower"),
            cageAngle, tiltAngle);

    // process setup
    ps::gpu::Process<NumericType, D> process(context, geometry, model,
                                             timePerAngle);
    process.setNumberOfRaysPerPoint(params.get<int>("raysPerPoint"));
    process.setIntegrationScheme(
        viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER);
    // process.setSmoothFlux(params.get("smoothFlux"));
    // process.setTimeStepRatio(params.get("timeStepRatio"));

    // run the process
    process.apply();

    // save the surface mesh
    auto copy = ps::SmartPointer<ps::Domain<NumericType, D>>::New(geometry);
    clean<NumericType, D>(copy, params.get("smoothingSize"));
    {
      ps::RayTracingParameters<NumericType, D> rayParams;
      rayParams.raysPerPoint = params.get<int>("raysPerPoint") * 2;
      rayParams.smoothingNeighbors = 2;
      ps::gpu::Process<NumericType, D> visProcess(context, copy, model, 0.);
      visProcess.setRayTracingParameters(rayParams);

      auto fluxMesh = visProcess.calculateFlux();

      auto surfMesh = ps::SmartPointer<ls::Mesh<float>>::New();
      ps::gpu::CreateSurfaceMesh<NumericType, float, D>(
          copy->getLevelSets().back(), surfMesh)
          .apply();

      ps::KDTree<NumericType, ps::Vec3D<NumericType>> pointKdTree;
      pointKdTree.setPoints(fluxMesh->nodes);
      pointKdTree.build();

      FluxOnMesh<NumericType>(fluxMesh->getCellData(), pointKdTree, surfMesh);
      viennals::VTKWriter<float>(surfMesh,
                                 params.get<std::string>("outputFile") + "_T" +
                                     std::to_string(counter) + ".vtp")
          .apply();
      copy->saveHullMesh(params.get<std::string>("outputFile") + "_T" +
                             std::to_string(counter++),
                         1e-1);
    }

    time += timePerAngle;
    cageAngle += 45.;
    cageAngle = std::fmod(cageAngle, 360.);

    std::cout << "Time: " << time << std::endl;
  }

  clean<NumericType, D>(geometry, params.get("smoothingSize"));

  // print final surface
  geometry->saveSurfaceMesh(params.get<std::string>("outputFile") + ".vtk");
  geometry->saveVolumeMesh(params.get<std::string>("outputFile"));
  geometry->saveLevelSets(params.get<std::string>("outputFile"));
}
