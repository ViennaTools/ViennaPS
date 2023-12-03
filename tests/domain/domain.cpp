#include <psDomain.hpp>
#include <psTestAssert.hpp>

#include <lsMakeGeometry.hpp>

int main() {
  constexpr int D = 2;

  {
    // default constructor
    auto domain = psSmartPointer<psDomain<double, D>>::New();
    PSTEST_ASSERT(domain);
  }

  {
    // single LS constructor
    auto ls = psSmartPointer<lsDomain<double, D>>::New();
    auto domain = psSmartPointer<psDomain<double, D>>::New(ls);
    PSTEST_ASSERT(domain);
  }

  {
    // two plane geometries
    lsBoundaryConditionEnum<D> boundaryCondition[D] = {
        lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY,
        lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY};
    double bounds[2 * D] = {-1., 1., -1., 1.};

    double origin[D] = {0.};
    double normal[D] = {0.};
    normal[D - 1] = 1.;

    auto plane1 = lsSmartPointer<lsDomain<double, D>>::New(
        bounds, boundaryCondition, 0.2);
    lsMakeGeometry<double, D>(
        plane1, lsSmartPointer<lsPlane<double, D>>::New(origin, normal))
        .apply();

    origin[D - 1] = 1.;
    auto plane2 = lsSmartPointer<lsDomain<double, D>>::New(
        bounds, boundaryCondition, 0.2);
    lsMakeGeometry<double, D>(
        plane2, lsSmartPointer<lsPlane<double, D>>::New(origin, normal))
        .apply();

    auto levelSets =
        psSmartPointer<std::vector<psSmartPointer<lsDomain<double, D>>>>::New();
    levelSets->push_back(plane1);
    levelSets->push_back(plane2);

    auto domain =
        psSmartPointer<psDomain<double, D>>::New(levelSets, true, 3., true);
    PSTEST_ASSERT(domain->getLevelSets()->size() == 2);
    PSTEST_ASSERT(domain->getCellSet());

    auto cellSet = domain->getCellSet();
    PSTEST_ASSERT(cellSet);
    PSTEST_ASSERT(cellSet->getDepth() == 3.);
    PSTEST_ASSERT(cellSet->getNumberOfCells() == 160);

    domain->clear();
    PSTEST_ASSERT(domain->getLevelSets()->size() == 0);

    // insert level sets
    domain->insertNextLevelSet(plane1);
    PSTEST_ASSERT(domain->getLevelSets()->size() == 1);

    domain->clear();
    domain->insertNextLevelSetAsMaterial(plane1, psMaterial::Si);
    PSTEST_ASSERT(domain->getLevelSets()->size() == 1);
    PSTEST_ASSERT(domain->getMaterialMap());

    // deep copy
    domain->insertNextLevelSetAsMaterial(plane2, psMaterial::SiO2);
    domain->generateCellSet(3., true);

    auto domainCopy = psSmartPointer<psDomain<double, D>>::New();
    domainCopy->deepCopy(domain);
    PSTEST_ASSERT(domainCopy->getLevelSets());
    PSTEST_ASSERT(domainCopy->getLevelSets()->size() == 2);
    PSTEST_ASSERT(domainCopy->getMaterialMap());
    PSTEST_ASSERT(domainCopy->getCellSet());

    // PSTEST_ASSERT deep copy
    PSTEST_ASSERT(domainCopy->getLevelSets().get() !=
                  domain->getLevelSets().get());
    PSTEST_ASSERT(domainCopy->getCellSet().get() != domain->getCellSet().get());
    PSTEST_ASSERT(domainCopy->getMaterialMap().get() !=
                  domain->getMaterialMap().get());
  }
}