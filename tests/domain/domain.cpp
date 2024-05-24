#include <lsMakeGeometry.hpp>
#include <psDomain.hpp>
#include <vcTestAsserts.hpp>

namespace viennacore {

namespace ps = viennaps;

template <class NumericType, int D> void RunTest() {
  {
    // default constructor
    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
    VC_TEST_ASSERT(domain);
  }

  {
    // single LS constructor
    auto ls = ps::SmartPointer<lsDomain<NumericType, D>>::New();
    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(ls);
    VC_TEST_ASSERT(domain);
  }

  {
    // two plane geometries
    lsBoundaryConditionEnum<D> boundaryCondition[D] = {
        lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
        lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY};
    double bounds[2 * D] = {-1., 1., -1., 1.};

    NumericType origin[D] = {0.};
    NumericType normal[D] = {0.};
    normal[D - 1] = 1.;

    auto plane1 = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCondition, 0.2);
    lsMakeGeometry<NumericType, D>(
        plane1, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    origin[D - 1] = 1.;
    auto plane2 = lsSmartPointer<lsDomain<NumericType, D>>::New(
        bounds, boundaryCondition, 0.2);
    lsMakeGeometry<NumericType, D>(
        plane2, lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal))
        .apply();

    auto levelSets = ps::SmartPointer<
        std::vector<ps::SmartPointer<lsDomain<NumericType, D>>>>::New();
    levelSets->push_back(plane1);
    levelSets->push_back(plane2);

    auto domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(levelSets);
    domain->generateCellSet(3., ps::Material::GAS, true);
    VC_TEST_ASSERT(domain->getLevelSets()->size() == 2);
    VC_TEST_ASSERT(domain->getCellSet());

    auto cellSet = domain->getCellSet();
    VC_TEST_ASSERT(cellSet);
    VC_TEST_ASSERT(cellSet->getDepth() == 3.);
    VC_TEST_ASSERT(cellSet->getNumberOfCells() == 160);

    domain->clear();
    VC_TEST_ASSERT(domain->getLevelSets()->size() == 0);

    // insert level sets
    domain->insertNextLevelSet(plane1);
    VC_TEST_ASSERT(domain->getLevelSets()->size() == 1);

    domain->clear();
    domain->insertNextLevelSetAsMaterial(plane1, ps::Material::Si);
    VC_TEST_ASSERT(domain->getLevelSets()->size() == 1);
    VC_TEST_ASSERT(domain->getMaterialMap());

    // deep copy
    domain->insertNextLevelSetAsMaterial(plane2, ps::Material::SiO2);
    domain->generateCellSet(3., ps::Material::GAS, true);

    auto domainCopy = ps::SmartPointer<ps::Domain<NumericType, D>>::New();
    domainCopy->deepCopy(domain);
    VC_TEST_ASSERT(domainCopy->getLevelSets());
    VC_TEST_ASSERT(domainCopy->getLevelSets()->size() == 2);
    VC_TEST_ASSERT(domainCopy->getMaterialMap());
    VC_TEST_ASSERT(domainCopy->getCellSet());

    // VC_TEST_ASSERT deep copy
    VC_TEST_ASSERT(domainCopy->getLevelSets().get() !=
                   domain->getLevelSets().get());
    VC_TEST_ASSERT(domainCopy->getCellSet().get() !=
                   domain->getCellSet().get());
    VC_TEST_ASSERT(domainCopy->getMaterialMap().get() !=
                   domain->getMaterialMap().get());
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }