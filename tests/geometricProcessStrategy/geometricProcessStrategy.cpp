#include <process/psGeometricModel.hpp>
#include <process/psGeometricProcessStrategy.hpp>
#include <process/psProcessModel.hpp>

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

// Mock process model for testing
template <typename NumericType, int D>
class MockGeometricProcessModel : public ProcessModelBase<NumericType, D> {
public:
  MockGeometricProcessModel() = default;

  MockGeometricProcessModel(
      SmartPointer<GeometricModel<NumericType, D>> geomModel) {
    this->setGeometricModel(geomModel);
  }
};

// Helper function to create a basic process context
template <typename NumericType, int D>
ProcessContext<NumericType, D> createBasicGeometricContext(
    SmartPointer<Domain<NumericType, D>> domain,
    SmartPointer<ProcessModelBase<NumericType, D>> model) {

  ProcessContext<NumericType, D> context;
  context.domain = domain;
  context.model = model;
  context.processDuration =
      0.0; // Geometric processes typically have zero duration
  context.processTime = 0.0;
  context.timeStep = 0.0;
  context.currentIteration = 0;

  // Set flags for geometric process
  context.flags.isGeometric = true;
  context.flags.useFluxEngine = false;
  context.flags.useAdvectionCallback = false;

  return context;
}

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  // Test 1: Strategy canHandle method - should reject non-geometric processes
  {
    auto domain = Domain<NumericType, D>::New();
    auto geomModel = SmartPointer<GeometricModel<NumericType, D>>::New();
    auto model =
        SmartPointer<MockGeometricProcessModel<NumericType, D>>::New(geomModel);
    auto context = createBasicGeometricContext<NumericType, D>(domain, model);
    context.flags.isGeometric = false; // Not geometric

    GeometricProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 2: Strategy canHandle method - should accept geometric processes
  {
    auto domain = Domain<NumericType, D>::New();
    auto geomModel = SmartPointer<GeometricModel<NumericType, D>>::New();
    auto model =
        SmartPointer<MockGeometricProcessModel<NumericType, D>>::New(geomModel);
    auto context = createBasicGeometricContext<NumericType, D>(domain, model);

    GeometricProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(strategy.canHandle(context));
  }

  // Test 3: Strategy canHandle method - should accept geometric process even
  // with other flags set
  {
    auto domain = Domain<NumericType, D>::New();
    auto geomModel = SmartPointer<GeometricModel<NumericType, D>>::New();
    auto model =
        SmartPointer<MockGeometricProcessModel<NumericType, D>>::New(geomModel);
    auto context = createBasicGeometricContext<NumericType, D>(domain, model);
    context.flags.useFluxEngine = true; // Other flags shouldn't matter
    context.flags.useAdvectionCallback = true;

    GeometricProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(
        strategy.canHandle(context)); // Only isGeometric flag matters
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
