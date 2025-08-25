#include <process/psAnalyticProcessStrategy.hpp>
#include <psProcessModelBase.hpp>
#include <psVelocityField.hpp>

#include <geometries/psMakePlane.hpp>
#include <psDomain.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

// Mock velocity field for testing
template <typename NumericType, int D>
class MockVelocityField : public VelocityField<NumericType, D> {
private:
  NumericType rate_;
  bool prepareCalled = false;

public:
  explicit MockVelocityField(NumericType rate = 1.0) : rate_(rate) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    return rate_;
  }

  int getTranslationFieldOptions() const override { return 0; }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType processTime) override {
    prepareCalled = true;
  }

  bool wasPrepareCalled() const { return prepareCalled; }
  void resetPrepareFlag() { prepareCalled = false; }
};

// Mock callback for testing
template <typename NumericType, int D>
class MockAdvectionCallback : public AdvectionCallback<NumericType, D> {
private:
  bool preAdvectCalled = false;
  bool postAdvectCalled = false;
  NumericType lastPreAdvectTime = 0.0;
  NumericType lastPostAdvectTime = 0.0;

public:
  bool applyPreAdvect(const NumericType processTime) override {
    preAdvectCalled = true;
    lastPreAdvectTime = processTime;
    return true;
  }

  bool applyPostAdvect(const NumericType advectionTime) override {
    postAdvectCalled = true;
    lastPostAdvectTime = advectionTime;
    return true;
  }

  bool wasPreAdvectCalled() const { return preAdvectCalled; }
  bool wasPostAdvectCalled() const { return postAdvectCalled; }
  NumericType getLastPreAdvectTime() const { return lastPreAdvectTime; }
  NumericType getLastPostAdvectTime() const { return lastPostAdvectTime; }

  void reset() {
    preAdvectCalled = false;
    postAdvectCalled = false;
    lastPreAdvectTime = 0.0;
    lastPostAdvectTime = 0.0;
  }
};

// Mock process model for testing
template <typename NumericType, int D>
class MockProcessModel : public ProcessModelBase<NumericType, D> {
public:
  MockProcessModel() = default;

  MockProcessModel(SmartPointer<VelocityField<NumericType, D>> velField) {
    this->setVelocityField(velField);
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();
    this->setSurfaceModel(surfModel);
  }

  MockProcessModel(SmartPointer<VelocityField<NumericType, D>> velField,
                   SmartPointer<AdvectionCallback<NumericType, D>> callback) {
    this->setVelocityField(velField);
    this->setAdvectionCallback(callback);
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();
    this->setSurfaceModel(surfModel);
  }
};

// Helper function to create a basic process context
template <typename NumericType, int D>
ProcessContext<NumericType, D>
createBasicContext(SmartPointer<Domain<NumericType, D>> domain,
                   SmartPointer<ProcessModelBase<NumericType, D>> model,
                   NumericType duration) {

  ProcessContext<NumericType, D> context;
  context.domain = domain;
  context.model = model;
  context.processDuration = duration;
  context.processTime = 0.0;
  context.timeStep = 0.0;
  context.currentIteration = 0;

  // Set flags based on model
  context.flags.isGeometric = false;
  context.flags.useFluxEngine = false;
  context.flags.useAdvectionCallback =
      (model->getAdvectionCallback() != nullptr);

  // Set default advection parameters
  context.advectionParams.integrationScheme =
      viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  context.advectionParams.timeStepRatio = 0.4999;
  context.advectionParams.velocityOutput = false;
  context.advectionParams.dissipationAlpha = 0.0;
  context.advectionParams.ignoreVoids = false;
  context.advectionParams.checkDissipation = false;

  return context;
}

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);

  // Test 1: Strategy canHandle method - should reject context without velocity
  // field
  {
    auto domain = Domain<NumericType, D>::New();
    auto model =
        SmartPointer<MockProcessModel<NumericType, D>>::New(); // No velocity
                                                               // field
    auto context = createBasicContext<NumericType, D>(domain, model, 1.0);

    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 2: Strategy canHandle method - should reject geometric processes
  {
    auto domain = Domain<NumericType, D>::New();
    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(1.0);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context = createBasicContext<NumericType, D>(domain, model, 1.0);
    context.flags.isGeometric = true;

    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 3: Strategy canHandle method - should reject flux engine processes
  {
    auto domain = Domain<NumericType, D>::New();
    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(1.0);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context = createBasicContext<NumericType, D>(domain, model, 1.0);
    context.flags.useFluxEngine = true;

    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 4: Strategy canHandle method - should reject zero duration
  {
    auto domain = Domain<NumericType, D>::New();
    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(1.0);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context =
        createBasicContext<NumericType, D>(domain, model, 0.0); // Zero duration

    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(!strategy.canHandle(context));
  }

  // Test 5: Strategy canHandle method - should accept valid analytic process
  {
    auto domain = Domain<NumericType, D>::New();
    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(1.0);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context = createBasicContext<NumericType, D>(domain, model, 1.0);

    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(strategy.canHandle(context));
  }

  // Test 6: Strategy execute method - should fail with invalid context (no
  // velocity field)
  {
    auto domain = Domain<NumericType, D>::New();
    auto model =
        SmartPointer<MockProcessModel<NumericType, D>>::New(); // No velocity
                                                               // field
    auto context = createBasicContext<NumericType, D>(domain, model, 1.0);

    AnalyticProcessStrategy<NumericType, D> strategy;
    auto result = strategy.execute(context);
    VC_TEST_ASSERT(result == ProcessResult::INVALID_INPUT);
  }

  // Test 7: Strategy execute method - basic execution without callbacks
  {
    auto domain = Domain<NumericType, D>::New(1.0, 10., 10.);
    MakePlane<NumericType, D>(domain).apply();

    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(-0.1);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context = createBasicContext<NumericType, D>(domain, model,
                                                      0.05); // Small duration

    AnalyticProcessStrategy<NumericType, D> strategy;

    // Verify strategy can handle this context
    VC_TEST_ASSERT(strategy.canHandle(context));

    // Execute the strategy
    auto result = strategy.execute(context);

    // Verify execution was successful
    VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

    // Verify the context was updated
    VC_TEST_ASSERT(context.processTime > 0.0);
    VC_TEST_ASSERT(context.currentIteration > 0);

    // Verify velocity field was prepared
    VC_TEST_ASSERT(velField->wasPrepareCalled());
  }

  // Test 8: Strategy execute method - execution with callbacks
  {
    auto domain = Domain<NumericType, D>::New(1.0, 10., 10.);
    MakePlane<NumericType, D>(domain).apply();

    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(-0.1);
    auto callback = SmartPointer<MockAdvectionCallback<NumericType, D>>::New();
    auto model =
        SmartPointer<MockProcessModel<NumericType, D>>::New(velField, callback);
    auto context = createBasicContext<NumericType, D>(domain, model, 0.05);

    AnalyticProcessStrategy<NumericType, D> strategy;

    // Verify callbacks haven't been called yet
    VC_TEST_ASSERT(!callback->wasPreAdvectCalled());
    VC_TEST_ASSERT(!callback->wasPostAdvectCalled());

    // Execute the strategy
    auto result = strategy.execute(context);

    // Verify execution was successful
    VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

    // Verify callbacks were called
    VC_TEST_ASSERT(callback->wasPreAdvectCalled());
    VC_TEST_ASSERT(callback->wasPostAdvectCalled());

    // Verify context was updated
    VC_TEST_ASSERT(context.processTime > 0.0);
    VC_TEST_ASSERT(context.currentIteration > 0);
  }

  // Test 9: Strategy execute method - callback returning false (early
  // termination)
  {
    // Create a callback that returns false to test early termination
    class FailingCallback : public AdvectionCallback<NumericType, D> {
    public:
      bool applyPreAdvect(const NumericType processTime) override {
        return false; // Fail immediately
      }
      bool applyPostAdvect(const NumericType advectionTime) override {
        return true;
      }
    };

    auto domain = Domain<NumericType, D>::New(1.0, 10., 10.);
    MakePlane<NumericType, D>(domain).apply();

    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(-0.1);
    auto callback = SmartPointer<FailingCallback>::New();
    auto model =
        SmartPointer<MockProcessModel<NumericType, D>>::New(velField, callback);
    auto context = createBasicContext<NumericType, D>(domain, model, 0.1);

    AnalyticProcessStrategy<NumericType, D> strategy;

    auto result = strategy.execute(context);

    // Should terminate early due to callback failure
    VC_TEST_ASSERT(result == ProcessResult::EARLY_TERMINATION);
  }

  // Test 10: Test strategy name and class identification
  {
    AnalyticProcessStrategy<NumericType, D> strategy;
    VC_TEST_ASSERT(strategy.name() == "AnalyticProcessStrategy");
  }

  // Test 11: Test context modification during execution
  {
    auto domain = Domain<NumericType, D>::New(1.0, 10., 10.);
    MakePlane<NumericType, D>(domain).apply();

    auto velField = SmartPointer<MockVelocityField<NumericType, D>>::New(-0.1);
    auto model = SmartPointer<MockProcessModel<NumericType, D>>::New(velField);
    auto context = createBasicContext<NumericType, D>(domain, model, 0.1);

    // Store initial values
    auto initialProcessTime = context.processTime;
    auto initialIteration = context.currentIteration;

    AnalyticProcessStrategy<NumericType, D> strategy;

    auto result = strategy.execute(context);

    VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

    // Verify context was properly modified
    VC_TEST_ASSERT(context.processTime > initialProcessTime);
    VC_TEST_ASSERT(context.currentIteration > initialIteration);
    VC_TEST_ASSERT(context.processTime >= context.processDuration);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
