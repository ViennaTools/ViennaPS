#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

namespace ps = viennaps;
namespace ls = viennals;

template <class NumericType, int D> void RunTest() {
  using lsDomainType = SmartPointer<ls::Domain<NumericType, D>>;
  using psDomainType = SmartPointer<ps::Domain<NumericType, D>>;

  {
    // default constructor
    auto domain = psDomainType::New();
    VC_TEST_ASSERT(domain);
  }

  {
    // single LS constructor
    auto ls = lsDomainType::New();
    auto domain = psDomainType::New(ls);
    VC_TEST_ASSERT(domain);
  }

  {
    // two plane geometries
    ls::BoundaryConditionEnum boundaryCondition[D];
    double bounds[2 * D];

    for (int i = 0; i < D; ++i) {
      bounds[2 * i] = -1.;
      bounds[2 * i + 1] = 1.;
      boundaryCondition[i] = ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY;
    }
    boundaryCondition[D - 1] = ls::BoundaryConditionEnum::INFINITE_BOUNDARY;

    NumericType origin[D] = {0.};
    NumericType normal[D] = {0.};
    normal[D - 1] = 1.;

    auto plane1 = lsDomainType::New(bounds, boundaryCondition, 0.2);
    ls::MakeGeometry<NumericType, D>(
        plane1, SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    origin[D - 1] = 1.;
    auto plane2 = lsDomainType::New(bounds, boundaryCondition, 0.2);
    ls::MakeGeometry<NumericType, D>(
        plane2, SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    std::vector<lsDomainType> levelSets;
    levelSets.push_back(plane1);
    levelSets.push_back(plane2);

    auto domain = psDomainType::New(levelSets);
    domain->generateCellSet(3., ps::Material::GAS, true);
    VC_TEST_ASSERT(domain->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domain->getCellSet());

    auto cellSet = domain->getCellSet();
    VC_TEST_ASSERT(cellSet);
    VC_TEST_ASSERT(cellSet->getDepth() == 3.);

    domain->clear();
    VC_TEST_ASSERT(domain->getLevelSets().size() == 0);

    // insert level sets
    domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
    VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
    VC_TEST_ASSERT(domain->getMaterialMap());

    // deep copy
    domain->insertNextLevelSetAsMaterial(plane2, ps::Material::SiO2);
    domain->generateCellSet(3., ps::Material::GAS, true);

    auto domainCopy = psDomainType::New();
    domainCopy->deepCopy(domain);
    VC_TEST_ASSERT(domainCopy->getLevelSets().size() == 2);
    VC_TEST_ASSERT(domainCopy->getMaterialMap());
    VC_TEST_ASSERT(domainCopy->getCellSet());

    // VC_TEST_ASSERT deep copy
    VC_TEST_ASSERT(domainCopy->getCellSet().get() !=
                   domain->getCellSet().get());
    VC_TEST_ASSERT(domainCopy->getMaterialMap().get() !=
                   domain->getMaterialMap().get());
  }

  // remove level sets
  {
    // two plane geometries
    ls::BoundaryConditionEnum boundaryCondition[D];
    double bounds[2 * D];

    for (int i = 0; i < D; ++i) {
      bounds[2 * i] = -1.;
      bounds[2 * i + 1] = 1.;
      boundaryCondition[i] = ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY;
    }
    boundaryCondition[D - 1] = ls::BoundaryConditionEnum::INFINITE_BOUNDARY;

    NumericType origin[D] = {0.};
    NumericType normal[D] = {0.};
    normal[D - 1] = 1.;

    auto plane1 = lsDomainType::New(bounds, boundaryCondition, 0.2);
    ls::MakeGeometry<NumericType, D>(
        plane1, SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    origin[D - 1] = 1.;
    auto plane2 = lsDomainType::New(bounds, boundaryCondition, 0.2);
    ls::MakeGeometry<NumericType, D>(
        plane2, SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
        .apply();

    {
      auto domain = psDomainType::New();
      domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
      domain->insertNextLevelSetAsMaterial(plane2, ps::Material::SiO2);

      domain->removeTopLevelSet();
      VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
    }

    {
      auto domain = psDomainType::New();
      domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
      domain->insertNextLevelSetAsMaterial(plane2, ps::Material::SiO2);

      domain->removeLevelSet(1);
      VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
    }

    {
      auto domain = psDomainType::New();
      domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
      domain->insertNextLevelSetAsMaterial(plane2, ps::Material::SiO2);

      domain->removeMaterial(ps::Material::Si);
      VC_TEST_ASSERT(domain->getLevelSets().size() == 1);
    }

    {
      auto domain = psDomainType::New();
      domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
      domain->insertNextLevelSetAsMaterial(plane2, ps::Material::Si);

      domain->removeMaterial(ps::Material::Si);
      VC_TEST_ASSERT(domain->getLevelSets().empty());
    }
  }
}

} // namespace viennacore

int main() {
  std::cout << "Running ViennaPS version: " << viennaps::version << std::endl;
  std::cout << "Major: " << viennaps::versionMajor
            << ", Minor: " << viennaps::versionMinor
            << ", Patch: " << viennaps::versionPatch << std::endl;
  VC_RUN_ALL_TESTS
}