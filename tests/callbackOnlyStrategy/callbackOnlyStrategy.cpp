#include <process/psCallbackOnlyStrategy.hpp>
#include <process/psProcessModel.hpp>

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

// Mock callback class for testing
template <typename NumericType, int D>
class MockAdvectionCallback : public AdvectionCallback<NumericType, D> {
private:
  bool preAdvectCalled = false;
  NumericType lastProcessTime = 0.0;

public:
  bool applyPreAdvect(const NumericType processTime) override {
    preAdvectCalled = true;
    lastProcessTime = processTime;
    return true;
  }

  bool wasPreAdvectCalled() const { return preAdvectCalled; }
  NumericType getLastProcessTime() const { return lastProcessTime; }

  void reset() {
    preAdvectCalled = false;
    lastProcessTime = 0.0;
  }
};

// Mock process model for testing
template <typename NumericType, int D>
class MockProcessModel : public ProcessModelBase<NumericType, D> {
public:
  MockProcessModel() = default;

  MockProcessModel(SmartPointer<AdvectionCallback<NumericType, D>> callback) {
    this->setAdvectionCallback(callback);
  }
};

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  // Test 1: Strategy should not handle context without callback
  {
    auto domain = Domain<NumericType, D>::New();
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New();

    ProcessContext<NumericType, D> context;
    context.domain = domain;
    context.model = model;
    context.processDuration = 0.0;
    context.flags.useAdvectionCallback = false;

    CallbackOnlyStrategy<NumericType, D> strategy;

    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 2: Strategy should not handle context with non-zero duration
  {
    auto domain = Domain<NumericType, D>::New();
    auto callback = SmartPointer<MockAdvectionCallback<NumericType, D>>::New();
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(callback);

    ProcessContext<NumericType, D> context;
    context.domain = domain;
    context.model = model;
    context.processDuration = 1.0; // Non-zero duration
    context.flags.useAdvectionCallback = true;

    CallbackOnlyStrategy<NumericType, D> strategy;

    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 3: Strategy should handle context with callback and zero duration
  {
    auto domain = Domain<NumericType, D>::New();
    auto callback = SmartPointer<MockAdvectionCallback<NumericType, D>>::New();
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(callback);

    ProcessContext<NumericType, D> context;
    context.domain = domain;
    context.model = model;
    context.processDuration = 0.0;
    context.flags.useAdvectionCallback = true;

    CallbackOnlyStrategy<NumericType, D> strategy;

    VC_TEST_ASSERT(strategy.canHandle(context));
  }

  // Test 4: Execute strategy and verify callback is called
  {
    // Create a simple domain with a plane geometry
    auto domain = Domain<NumericType, D>::New(1., 10., 10.);
    MakePlane<NumericType, D>(domain).apply();

    auto callback = SmartPointer<MockAdvectionCallback<NumericType, D>>::New();
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(callback);

    ProcessContext<NumericType, D> context;
    context.domain = domain;
    context.model = model;
    context.processDuration = 0.0;
    context.flags.useAdvectionCallback = true;

    CallbackOnlyStrategy<NumericType, D> strategy;

    // Verify callback hasn't been called yet
    VC_TEST_ASSERT(!callback->wasPreAdvectCalled());

    // Execute the strategy
    auto result = strategy.execute(context);

    // Verify execution was successful
    VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

    // Verify callback was called with correct parameters
    VC_TEST_ASSERT(callback->wasPreAdvectCalled());
    VC_TEST_ASSERT(callback->getLastProcessTime() == 0.0);

    // Verify domain was set on callback (indirectly through successful
    // execution)
    VC_TEST_ASSERT(domain->getLevelSets().size() >= 1);
  }

  // Test 5: Strategy name is correct
  {
    CallbackOnlyStrategy<NumericType, D> strategy;
    assert(strategy.name() == "CallbackOnlyStrategy");
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
