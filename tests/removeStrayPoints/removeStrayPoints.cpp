#include <viennaps.hpp>

#include <vcTestAsserts.hpp>

using namespace viennaps;

int main() {
  Logger::setLogLevel(LogLevel::WARNING);
  using NumericType = double;
  constexpr int D = 2;

  auto domain = Domain<NumericType, D>::New(0.1, 10., 10.);
  MakeTrench<NumericType, D>(domain, 5.0, 0.0, 0.0, 1.0).apply();

  {
    auto isoProcess = SmartPointer<IsotropicProcess<NumericType, D>>::New(
        -1.0, Material::Mask);
    Process<NumericType, D>(domain, isoProcess, 1.0).apply();
  }

  {
    domain->duplicateTopLevelSet(Material::SiO2);
    auto isoProcess = SmartPointer<IsotropicProcess<NumericType, D>>::New(0.3);
    Process<NumericType, D>(domain, isoProcess, 1.0).apply();

    auto dirProcess = SmartPointer<DirectionalProcess<NumericType, D>>::New(
        Vec3D<NumericType>{0., 1., 0.}, .5);
    Process<NumericType, D>(domain, dirProcess, 1.0).apply();
  }

  {
    auto isoProcess = SmartPointer<IsotropicProcess<NumericType, D>>::New(
        -1.0, Material::SiO2);
    Process<NumericType, D>(domain, isoProcess, 1.0).apply();
  }

  domain->removeTopLevelSet();

  domain->saveSurfaceMesh("beforeRemovingStrayPoints");
  domain->saveLevelSetMesh("beforeRemovingStrayPoints");

  std::cout << "Number of components before removing stray points: "
            << domain->getNumberOfComponents() << std::endl;
  VC_TEST_ASSERT(domain->getNumberOfComponents() == 8);

  domain->removeStrayPoints();

  domain->saveSurfaceMesh("afterRemovingStrayPoints");
  domain->saveLevelSetMesh("afterRemovingStrayPoints");

  std::cout << "Number of components after removing stray points: "
            << domain->getNumberOfComponents() << std::endl;
  VC_TEST_ASSERT(domain->getNumberOfComponents() == 2);

  MakeTrench<NumericType, D>(domain, 5.0, 0.0, 0.0, 1.0).apply();
  domain->removeMaterial(Material::Si);

  VC_TEST_ASSERT(domain->getNumberOfComponents() == 3);

  domain->saveLevelSetMesh("testInitial");
}