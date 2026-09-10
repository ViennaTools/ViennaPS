#include <geometries/psMakeStack.hpp>
#include <models/psIsotropicProcess.hpp>
#include <process/psProcess.hpp>
#include <psCreateSurfaceMesh.hpp>
#include <psElementToPointData.hpp>

#include <nanoflann.hpp>

constexpr int D = 3;
using namespace viennaps;

struct NFPointCloud {
  std::vector<Vec3D<double>> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return pts[idx][dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }
};

double median(std::vector<double> vec) {
  size_t size = vec.size();
  std::sort(vec.begin(), vec.end());
  size_t mid = size / 2;
  return size % 2 == 0 ? (vec[mid - 1] + vec[mid]) / 2 : vec[mid];
}

auto makeGeometry(int i) {
  Logger::setLogLevel("ERROR");
  using NumericType = double;
  const NumericType gridDelta = 1. / NumericType(i);
  auto domain = Domain<NumericType, D>::New(gridDelta, 100., 100.,
                                            BoundaryType::REFLECTIVE_BOUNDARY);
  MakeStack<NumericType, D>(domain, 15, 10., 10.0, 0.0, 50.0, 0.0).apply();

  auto etch = SmartPointer<IsotropicProcess<NumericType, D>>::New(
      -1.0, Material::Si3N4);
  Process<NumericType, D>(domain, etch, 10).apply();

  domain->saveSurfaceMesh("stack");

  return domain;
}

int main() {
  Timer<> timer;

  int i = 2;
  auto domain = makeGeometry(i);
  const double gridDelta = 1. / double(i);
  const double searchRadius = gridDelta * 2.0;

  std::vector<std::string> dataLabels = {"data"};
  auto pointData = PointData<double>::New();
  auto diskMesh = domain->getDiskMesh();
  auto surfaceMesh = viennals::Mesh<double>::New();
  std::vector<Vec3D<double>> elementCenters;
  std::vector<std::vector<double>> elementDataArrays(1);

  // ViennaCore KDTree
  {
    CreateSurfaceMesh<double, double, D>(domain->getSurface(), surfaceMesh)
        .apply();

    for (const auto &cell : surfaceMesh->triangles) {
      Vec3D<double> center =
          (surfaceMesh->nodes[cell[0]] + surfaceMesh->nodes[cell[1]] +
           surfaceMesh->nodes[cell[2]]) /
          3.0;
      elementCenters.push_back(center);
    }

    timer.start();
    auto elementKdTree =
        SmartPointer<KDTree<double, Vec3D<double>>>::New(elementCenters);
    elementKdTree->build();
    timer.finish();

    std::cout << "ViennaCore KDTree build took " << timer.currentDuration * 1e-6
              << " ms" << std::endl;

    // generate random data for each element
    elementDataArrays[0].reserve(elementCenters.size());
    for (size_t i = 0; i < elementCenters.size(); ++i)
      elementDataArrays[0].push_back(static_cast<double>(rand()) / RAND_MAX);

    ElementToPointData<double, double, double, true, true,
                       KDTree<double, Vec3D<double>>>
        converter(dataLabels, pointData, elementKdTree, diskMesh, surfaceMesh,
                  gridDelta * 2.0);
    converter.setElementDataArrays(elementDataArrays);

    timer.start();
    converter.apply();
    timer.finish();

    auto conversionTime = timer.currentDuration * 1e-6;
    std::cout << "ViennaCore ElementToPointData conversion took "
              << conversionTime << " ms" << std::endl;
  }

  diskMesh->getCellData() = *pointData;
  viennals::VTKWriter<double>(diskMesh, "result_VC").apply();

  auto vc_result = *pointData;

  // nanoflann
  {
    pointData->clear();
    diskMesh = domain->getDiskMesh();

    PointCloud<double> cloud;
    cloud.positions = elementCenters;

    timer.start();
    auto elementKdTree =
        SmartPointer<NFKDTree<double, Vec3D<double>, 3>>::New(cloud);
    elementKdTree->build();
    timer.finish();

    std::cout << "NFKDTree build took " << timer.currentDuration * 1e-6 << " ms"
              << std::endl;

    ElementToPointData<double, double, double, true, true,
                       NFKDTree<double, Vec3D<double>, 3>>
        converter(dataLabels, pointData, elementKdTree, diskMesh, surfaceMesh,
                  gridDelta * 2.0);
    converter.setElementDataArrays(elementDataArrays);

    timer.start();
    converter.apply();
    timer.finish();

    auto conversionTime = timer.currentDuration * 1e-6;
    std::cout << "NF ElementToPointData conversion took " << conversionTime
              << " ms" << std::endl;
  }

  std::vector<double> differences;
  auto data = pointData->getScalarData(0);
  auto data_vc = vc_result.getScalarData(0);
  for (size_t i = 0; i < data_vc->size(); ++i)
    differences.push_back(std::abs(data_vc->at(i) - data->at(i)));
  pointData->insertNextScalarData(std::move(differences), "differences");

  diskMesh->getCellData() = *pointData;
  viennals::VTKWriter<double>(diskMesh, "result_NF").apply();

  auto centerMesh = viennals::Mesh<double>::New();
  centerMesh->nodes = elementCenters;
  unsigned id = 0;
  for (const auto &c : elementCenters) {
    id = centerMesh->insertNextVertex({id});
  }

  viennals::VTKWriter<double>(diskMesh, "diskMesh").apply();
  viennals::VTKWriter<double>(surfaceMesh, "surfaceMesh").apply();
  viennals::VTKWriter<double>(centerMesh, "centerMesh").apply();

  return 0;
}