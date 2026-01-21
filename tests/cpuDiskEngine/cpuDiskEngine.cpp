#include <process/psCPUDiskEngine.hpp>
#include <process/psProcessModel.hpp>
#include <process/psSurfaceModel.hpp>
#include <psDomain.hpp>

#include <geometries/psMakePlane.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaps;

// Mock surface model for testing
template <typename NumericType>
class MockSurfaceModel : public SurfaceModel<NumericType> {
private:
  bool initCoveragesCalled = false;
  bool initSurfaceDataCalled = false;
  bool calculateVelocitiesCalled = false;
  bool updateCoveragesCalled = false;

public:
  void initializeCoverages(unsigned numGeometryPoints) override {
    initCoveragesCalled = true;
    if (this->coverages == nullptr) {
      this->coverages = viennals::PointData<NumericType>::New();
    } else {
      this->coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints, 0.5);
    this->coverages->insertNextScalarData(cov, "testCoverage");
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    initSurfaceDataCalled = true;
    if (this->surfaceData == nullptr) {
      this->surfaceData = viennals::PointData<NumericType>::New();
    } else {
      this->surfaceData->clear();
    }
    std::vector<NumericType> data(numGeometryPoints, 2.0);
    this->surfaceData->insertNextScalarData(data, "testSurfaceData");
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    calculateVelocitiesCalled = true;
    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 1.0);
    return velocity;
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    updateCoveragesCalled = true;
    // Update test coverage based on flux
    auto coverage = this->coverages->getScalarData("testCoverage");
    for (size_t i = 0; i < coverage->size(); ++i) {
      coverage->at(i) = std::min(coverage->at(i) + 0.1, 1.0);
    }
  }

  // Test methods
  bool wasInitCoveragesCalled() const { return initCoveragesCalled; }
  bool wasInitSurfaceDataCalled() const { return initSurfaceDataCalled; }
  bool wasCalculateVelocitiesCalled() const {
    return calculateVelocitiesCalled;
  }
  bool wasUpdateCoveragesCalled() const { return updateCoveragesCalled; }

  void resetFlags() {
    initCoveragesCalled = false;
    initSurfaceDataCalled = false;
    calculateVelocitiesCalled = false;
    updateCoveragesCalled = false;
  }
};

// Mock particle for testing ray tracing
template <typename NumericType, int D>
class MockParticle
    : public viennaray::Particle<MockParticle<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        viennaray::RNG &rng) override final {
    // Simple collision model - add weight to flux
    localData.getVectorData(0)[primID] += rayWeight;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    viennaray::RNG &rng) override final {
    // Perfect sticking
    return std::pair<NumericType, Vec3D<NumericType>>{
        1.0, Vec3D<NumericType>{0., 0., 0.}};
  }

  void initNew(viennaray::RNG &rng) override final {}

  NumericType getSourceDistributionPower() const override final { return 1.0; }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {"testFlux"};
  }
};

// Mock process model for testing
template <typename NumericType, int D>
class MockProcessModel : public ProcessModelCPU<NumericType, D> {
public:
  MockProcessModel() {
    // Create mock surface model
    auto surfModel = SmartPointer<MockSurfaceModel<NumericType>>::New();
    this->setSurfaceModel(surfModel);

    // Create mock particle
    auto particle = std::make_unique<MockParticle<NumericType, D>>();
    this->insertNextParticleType(particle);

    this->setProcessName("MockProcessModel");
  }

  MockSurfaceModel<NumericType> *getMockSurfaceModel() const {
    return static_cast<MockSurfaceModel<NumericType> *>(
        this->getSurfaceModel().get());
  }

  bool useMaterialIds() const { return true; }
};

// Helper function to create a basic process context
template <typename NumericType, int D>
ProcessContext<NumericType, D> createBasicContext() {
  ProcessContext<NumericType, D> context;

  // Create simple domain with plane
  context.domain = SmartPointer<Domain<NumericType, D>>::New(1., 10., 10.);

  MakePlane<NumericType, D>(context.domain).apply();

  // Create mock process model
  context.model = SmartPointer<MockProcessModel<NumericType, D>>::New();

  // Set basic parameters
  context.processDuration = 1.0;
  context.processTime = 0.0;
  context.timeStep = 0.1;

  // Set ray tracing parameters
  context.rayTracingParams.raysPerPoint = 100;
  context.rayTracingParams.useRandomSeeds = true;
  context.rayTracingParams.diskRadius = 0.0;
  context.rayTracingParams.ignoreFluxBoundaries = false;
  context.rayTracingParams.smoothingNeighbors = 0;

  // Update flags based on model
  context.updateFlags();

  return context;
}

// Test cases
template <typename NumericType, int D> void test_checkInput() {
  std::cout << "Testing CPUDiskEngine::checkInput..." << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();

  // Test with valid process model
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test with invalid model (null)
  context.model = nullptr;
  result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::INVALID_INPUT);
}

template <typename NumericType, int D> void test_initialize() {
  std::cout << "Testing CPUDiskEngine::initialize..." << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test with ignore flux boundaries
  context.rayTracingParams.ignoreFluxBoundaries = true;
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test with random seeds disabled
  context.rayTracingParams.useRandomSeeds = false;
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test with different number of rays
  context.rayTracingParams.raysPerPoint = 500;
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
}

template <typename NumericType, int D> void test_updateSurface() {
  std::cout << "Testing CPUDiskEngine::updateSurface..." << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Initialize the engine first
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Create a disk mesh for the context
  context.diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> meshGenerator(context.diskMesh);

  for (auto ls : context.domain->getLevelSets()) {
    meshGenerator.insertNextLevelSet(ls);
  }
  meshGenerator.apply();

  // Test updateSurface with default disk radius
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test with custom disk radius
  context.rayTracingParams.diskRadius = 1.5;
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Reset disk radius
  context.rayTracingParams.diskRadius = 0.0;
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
}

template <typename NumericType, int D> void test_calculateFluxes() {
  std::cout << "Testing CPUDiskEngine::calculateFluxes..." << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Initialize the engine
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Create disk mesh
  context.diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> meshGenerator(context.diskMesh);

  for (auto ls : context.domain->getLevelSets()) {
    meshGenerator.insertNextLevelSet(ls);
  }
  meshGenerator.apply();

  // Update surface geometry
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test flux calculation
  auto fluxes = viennals::PointData<NumericType>::New();
  result = engine.calculateFluxes(context, fluxes);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
  VC_TEST_ASSERT(fluxes != nullptr);

  // Check that flux data was created
  VC_TEST_ASSERT(fluxes->getScalarDataSize() > 0);
  auto fluxData = fluxes->getScalarData("testFlux");
  VC_TEST_ASSERT(fluxData != nullptr);
  VC_TEST_ASSERT(fluxData->size() > 0);
}

template <typename NumericType, int D>
void test_calculateFluxesWithCoverages() {
  std::cout << "Testing CPUDiskEngine::calculateFluxes with coverages..."
            << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Enable coverages
  context.flags.useCoverages = true;

  // Initialize surface model with coverages
  auto mockModel =
      static_cast<MockProcessModel<NumericType, D> *>(context.model.get());
  auto surfModel = mockModel->getMockSurfaceModel();
  surfModel->initializeCoverages(10); // Initialize with some points

  // Initialize the engine
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Create disk mesh
  context.diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> meshGenerator(context.diskMesh);

  for (auto ls : context.domain->getLevelSets()) {
    meshGenerator.insertNextLevelSet(ls);
  }
  meshGenerator.apply();

  // Update surface geometry
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test flux calculation with coverages
  auto fluxes = viennals::PointData<NumericType>::New();
  result = engine.calculateFluxes(context, fluxes);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
  VC_TEST_ASSERT(fluxes != nullptr);
}

template <typename NumericType, int D> void test_fluxSmoothing() {
  std::cout << "Testing CPUDiskEngine flux smoothing..." << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Enable flux smoothing
  context.rayTracingParams.smoothingNeighbors = 2;

  // Initialize the engine
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Create disk mesh
  context.diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> meshGenerator(context.diskMesh);

  for (auto ls : context.domain->getLevelSets()) {
    meshGenerator.insertNextLevelSet(ls);
  }
  meshGenerator.apply();

  // Update surface geometry
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test flux calculation with smoothing
  auto fluxes = viennals::PointData<NumericType>::New();
  result = engine.calculateFluxes(context, fluxes);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
  VC_TEST_ASSERT(fluxes != nullptr);
}

template <typename NumericType, int D> void test_multipleParticleTypes() {
  std::cout << "Testing CPUDiskEngine with multiple particle types..."
            << std::endl;

  CPUDiskEngine<NumericType, D> engine;
  auto context = createBasicContext<NumericType, D>();
  auto result = engine.checkInput(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Add second particle type to model
  auto mockModel =
      static_cast<MockProcessModel<NumericType, D> *>(context.model.get());
  auto particle2 = std::make_unique<MockParticle<NumericType, D>>();
  mockModel->insertNextParticleType(particle2);

  // Initialize the engine
  result = engine.initialize(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Create disk mesh
  context.diskMesh = viennals::Mesh<NumericType>::New();
  viennals::ToDiskMesh<NumericType, D> meshGenerator(context.diskMesh);

  for (auto ls : context.domain->getLevelSets()) {
    meshGenerator.insertNextLevelSet(ls);
  }
  meshGenerator.apply();

  // Update surface geometry
  result = engine.updateSurface(context);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);

  // Test flux calculation with multiple particles
  auto fluxes = viennals::PointData<NumericType>::New();
  result = engine.calculateFluxes(context, fluxes);
  VC_TEST_ASSERT(result == ProcessResult::SUCCESS);
  VC_TEST_ASSERT(fluxes != nullptr);

  // Should have fluxes from both particle types
  VC_TEST_ASSERT(fluxes->getScalarDataSize() >= 2);
}

// Run all tests
template <typename NumericType, int D> void runAllTests() {
  std::cout << "Running CPUDiskEngine tests for " << D << "D..." << std::endl;

  test_checkInput<NumericType, D>();
  test_initialize<NumericType, D>();
  test_updateSurface<NumericType, D>();
  test_calculateFluxes<NumericType, D>();
  test_calculateFluxesWithCoverages<NumericType, D>();
  test_fluxSmoothing<NumericType, D>();
  test_multipleParticleTypes<NumericType, D>();

  std::cout << "All CPUDiskEngine tests passed for " << D << "D!" << std::endl;
}

} // namespace viennacore

int main() {
  using NumericType = double;

  try {
    viennacore::runAllTests<NumericType, 2>();
    viennacore::runAllTests<NumericType, 3>();
    std::cout << "All CPUDiskEngine tests completed successfully!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
