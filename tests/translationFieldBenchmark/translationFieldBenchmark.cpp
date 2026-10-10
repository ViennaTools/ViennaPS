#include <geometries/psMakeTrench.hpp>
#include <process/psTranslationField.hpp>
#include <psDomain.hpp>
#include <vcNFKDTree.hpp>
#include <vcTimer.hpp>

using namespace viennaps;

struct PointCloud {
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

int main() {
  using NumericType = double;
  constexpr int D = 3;
  using ConstSparseIterator = viennahrle::ConstSparseIterator<
      typename viennals::Domain<NumericType, D>::DomainType>;

  constexpr NumericType extent = 100.;
  constexpr NumericType trenchWidth = 50.;
  constexpr NumericType trenchDepth = 50.;

  auto makeGeometry = [&](int i) {
    const NumericType gridDelta = 1. / NumericType(i);
    auto domain = Domain<NumericType, D>::New(
        gridDelta, extent, extent, BoundaryType::REFLECTIVE_BOUNDARY);
    MakeTrench<NumericType, D>(domain, trenchWidth, trenchDepth).apply();
    return domain;
  };

  const int numRuns = 5;

  Timer<> timer;

  for (int i = 3; i < 8; ++i) {
    auto domain = makeGeometry(i);
    const NumericType gridDelta = 1. / NumericType(i);

    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToDiskMesh<NumericType, D> meshConverter;

    auto translator =
        SmartPointer<std::unordered_map<unsigned long, unsigned long>>::New();
    auto kdTree = SmartPointer<
        typename TranslationField<NumericType, D>::KDTreeType>::New();

    meshConverter.setMesh(mesh);
    meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    meshConverter.insertNextLevelSet(domain->getSurface());
    meshConverter.setTranslator(translator);

    timer.start();
    meshConverter.apply();
    timer.finish();
    auto mapBuildTime = timer.currentDuration * 1e-6;

    std::cout << "Mesh has " << mesh->getNodes().size() << " nodes."
              << std::endl;
    std::cout << "Mesh generation took " << mapBuildTime << " ms" << std::endl;

    auto tfTree = SmartPointer<TranslationField<NumericType, D>>::New(
        nullptr, domain->getMaterialMap(), 2);
    tfTree->setKdTree(kdTree);

    timer.start();
    tfTree->buildKdTree(mesh->getNodes());
    timer.finish();
    auto treeBuildTime = timer.currentDuration * 1e-6;

    std::cout << "KdTree build took " << treeBuildTime << " ms" << std::endl;

    auto tfMap = SmartPointer<TranslationField<NumericType, D>>::New(
        nullptr, domain->getMaterialMap(), 1);
    tfMap->setTranslator(translator);

    auto &hrleDomain = domain->getSurface()->getDomain();
    auto const &grid = domain->getGrid();
    viennahrle::Index<D> startVector = grid.getMinGridPoint();
    viennahrle::Index<D> endVector =
        grid.incrementIndices(grid.getMaxGridPoint());

    std::vector<double> times;
    for (int j = 0; j < numRuns; ++j) {
      timer.start();
      for (ConstSparseIterator it(hrleDomain, startVector);
           it.getStartIndices() < endVector; ++it) {

        if (!it.isDefined() || std::abs(it.getValue()) > 0.5)
          continue;

        const auto indices = it.getStartIndices();
        auto id = it.getPointId();

        Vec3D<NumericType> coords;
        for (unsigned i = 0; i < D; ++i) {
          coords[i] = indices[i] * gridDelta;
        }

        tfMap->translateLsId(id, coords);
      }
      timer.finish();
      times.push_back(timer.currentDuration * 1e-6);
    }

    auto timeMap = median(times);
    std::cout << "TranslationField with map took " << timeMap << " ms"
              << std::endl;
    std::cout << "total: " << timeMap + mapBuildTime << " ms" << std::endl;

    times.clear();
    for (int j = 0; j < numRuns; ++j) {
      timer.start();
      for (ConstSparseIterator it(hrleDomain, startVector);
           it.getStartIndices() < endVector; ++it) {

        if (!it.isDefined() || std::abs(it.getValue()) > 0.5)
          continue;

        const auto indices = it.getStartIndices();
        auto id = it.getPointId();

        Vec3D<NumericType> coords;
        for (unsigned i = 0; i < D; ++i) {
          coords[i] = indices[i] * gridDelta;
        }

        tfTree->translateLsId(id, coords);
      }
      timer.finish();
      times.push_back(timer.currentDuration * 1e-6);
    }

    auto timeTree = median(times);
    std::cout << "TranslationField with kdTree took " << timeTree << " ms"
              << std::endl;
    std::cout << "total: " << timeTree + treeBuildTime << " ms" << std::endl;

    std::cout << "Ratio (tree/map): "
              << static_cast<double>(timeTree + treeBuildTime) /
                     static_cast<double>(timeMap + mapBuildTime)
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }
}
