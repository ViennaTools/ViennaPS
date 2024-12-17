#pragma once

#include "context.hpp"
#include <cuda.h>
#include <optix_stubs.h>

#include <cstring>

#include <lsPointData.hpp>

#include <rayUtil.hpp>

#include <curtBoundary.hpp>
#include <curtGeometry.hpp>
#include <curtLaunchParams.hpp>
#include <curtParticle.hpp>
#include <curtSBTRecords.hpp>
#include <curtUtilities.hpp>

#include <utChecks.hpp>
#include <utCudaBuffer.hpp>
#include <utLaunchKernel.hpp>
#include <utLoadModules.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <class T, int D> class Trace {
public:
  /// constructor - performs all setup, including initializing
  /// optix, creates module, pipeline, programs, SBT, etc.
  Trace(Context passedContext) : context(passedContext) {
    initRayTracer();
    mesh = SmartPointer<viennals::Mesh<float>>::New();
  }

  void setKdTree(SmartPointer<KDTree<T, std::array<T, 3>>> passedKdTree) {
    kdTree = passedKdTree;
  }

  void setPipeline(std::string fileName,
                   std::string path = VIENNAPS_KERNELS_PATH) {
    // check if filename ends in .optixir
    if (fileName.find(".optixir") == std::string::npos) {
      if (fileName.find(".ptx") != std::string::npos)
        fileName += ".optixir";
    }

    if (!fileExists(path + fileName)) {
      Logger::getInstance()
          .addError("Pipeline file " + fileName + " not found.")
          .print();
    }

    pipelineName = fileName;
    pipelinePath = path;
  }

  void insertNextParticle(const Particle<T> &particle) {
    particles.push_back(particle);
  }

  void apply() {

    if (numCellData != 0 && cellDataBuffer.sizeInBytes == 0) {
      cellDataBuffer.allocInit(numCellData * launchParams.numElements, T(0));
    }
    assert(cellDataBuffer.sizeInBytes / sizeof(T) ==
           numCellData * launchParams.numElements);

    // resize our cuda result buffer
    resultBuffer.allocInit(launchParams.numElements * numRates, T(0));
    launchParams.resultBuffer = (T *)resultBuffer.dPointer();

    if (useRandomSeed) {
      std::random_device rd;
      std::uniform_int_distribution<unsigned int> gen;
      launchParams.seed = gen(rd);
    } else {
      launchParams.seed = runNumber++;
    }

    int numPointsPerDim =
        static_cast<int>(std::sqrt(static_cast<T>(launchParams.numElements)));

    if (numberOfRaysFixed > 0) {
      numPointsPerDim = 1;
      numberOfRaysPerPoint = numberOfRaysFixed;
    }

    numRays = numPointsPerDim * numPointsPerDim * numberOfRaysPerPoint;
    if (numRays > (1 << 29)) {
      Logger::getInstance()
          .addError("Too many rays for single launch: " +
                    util::prettyDouble(numRays))
          .print();
    }
    Logger::getInstance()
        .addDebug("Number of rays: " + util::prettyDouble(numRays))
        .print();

    for (size_t i = 0; i < particles.size(); i++) {

      launchParams.cosineExponent = particles[i].cosineExponent;
      launchParams.sticking = particles[i].sticking;
      launchParams.meanIonEnergy = particles[i].meanIonEnergy;
      launchParams.sigmaIonEnergy = particles[i].sigmaIonEnergy;
      launchParams.source.directionBasis =
          rayInternal::getOrthonormalBasis<float>(particles[i].direction);
      launchParamsBuffer.upload(&launchParams, 1);

      CUstream stream;
      CUDA_CHECK(StreamCreate(&stream));
      buildSBT(i);
      OPTIX_CHECK(optixLaunch(pipelines[i], stream,
                              /*! parameters and SBT */
                              launchParamsBuffer.dPointer(),
                              launchParamsBuffer.sizeInBytes, &sbts[i],
                              /*! dimensions of the launch: */
                              numPointsPerDim, numPointsPerDim,
                              numberOfRaysPerPoint));
    }

    // T *temp = new T[launchParams.numElements];
    // resultBuffer.download(temp, launchParams.numElements);
    // for (int i = 0; i < launchParams.numElements; i++) {
    //   std::cout << temp[i] << std::endl;
    // }
    // delete temp;
    // std::cout << util::prettyDouble(numRays * particles.size()) << std::endl;

    // sync - maybe remove in future
    cudaDeviceSynchronize();
    normalize();
  }

  // void translateToPointData(SmartPointer<viennals::Mesh<T>> mesh,
  //                           CudaBuffer &pointDataBuffer, T radius = 0.,
  //                           const bool download = false) {
  //   // upload oriented pointcloud data to device
  //   assert(mesh->nodes.size() != 0 &&
  //          "Passing empty mesh in translateToPointValuesSphere.");
  //   if (radius == 0.)
  //     radius = launchParams.source.gridDelta;
  //   size_t numValues = mesh->nodes.size();
  //   CudaBuffer pointBuffer;
  //   pointBuffer.allocUpload(mesh->nodes);
  //   pointDataBuffer.allocInit(numValues * numRates, T(0));

  //   CUdeviceptr d_vertex = geometry.geometryVertexBuffer.dPointer();
  //   CUdeviceptr d_index = geometry.geometryIndexBuffer.dPointer();
  //   CUdeviceptr d_values = resultBuffer.dPointer();
  //   CUdeviceptr d_point = pointBuffer.dPointer();
  //   CUdeviceptr d_pointValues = pointDataBuffer.dPointer();

  //   void *kernel_args[] = {
  //       &d_vertex,      &d_index, &d_values,  &d_point,
  //       &d_pointValues, &radius,  &numValues, &launchParams.numElements,
  //       &numRates};

  //   LaunchKernel::launch(translateModuleName, translateToPointDataKernelName,
  //                        kernel_args, context, sizeof(int));

  //   if (download) {
  //     downloadResultsToPointData(mesh->getCellData(), pointDataBuffer,
  //                                mesh->nodes.size());
  //   }

  //   pointBuffer.free();
  // }

  void setElementData(CudaBuffer &passedCellDataBuffer, unsigned numData) {
    assert(passedCellDataBuffer.sizeInBytes / sizeof(T) / numData ==
           launchParams.numElements);
    cellDataBuffer = passedCellDataBuffer;
  }

  // void translateFromPointData(SmartPointer<viennals::Mesh<T>> mesh,
  //                             CudaBuffer &pointDataBuffer, unsigned numData)
  //                             {
  //   // upload oriented pointcloud data to device
  //   size_t numPoints = mesh->nodes.size();
  //   assert(mesh->nodes.size() > 0);
  //   assert(pointDataBuffer.sizeInBytes / sizeof(T) / numData == numPoints);
  //   assert(numData > 0);

  //   CudaBuffer pointBuffer;
  //   pointBuffer.allocUpload(mesh->nodes);

  //   cellDataBuffer.alloc(launchParams.numElements * numData * sizeof(T));

  //   CUdeviceptr d_vertex = geometry.geometryVertexBuffer.dPointer();
  //   CUdeviceptr d_index = geometry.geometryIndexBuffer.dPointer();
  //   CUdeviceptr d_values = cellDataBuffer.dPointer();
  //   CUdeviceptr d_point = pointBuffer.dPointer();
  //   CUdeviceptr d_pointValues = pointDataBuffer.dPointer();

  //   void *kernel_args[] = {&d_vertex,
  //                          &d_index,
  //                          &d_values,
  //                          &d_point,
  //                          &d_pointValues,
  //                          &numPoints,
  //                          &launchParams.numElements,
  //                          &numData};

  //   LaunchKernel::launch(translateModuleName,
  //   translateFromPointDataKernelName,
  //                        kernel_args, context, sizeof(int));

  //   pointBuffer.free();
  // }

  void updateSurface() {
    geometry.buildAccelFromDomain(domain, launchParams, mesh, kdTree);
    geometryValid = true;
  }

  void setNumberOfRaysPerPoint(const size_t pNumRays) {
    numberOfRaysPerPoint = pNumRays;
  }

  void setFixedNumberOfRays(const size_t pNumRays) {
    numberOfRaysFixed = pNumRays;
  }

  void setUseRandomSeed(const bool set) { useRandomSeed = set; }

  void getFlux(T *flux, int particleIdx, int dataIdx) {
    unsigned int offset = 0;
    for (size_t i = 0; i < particles.size(); i++) {
      if (particleIdx > i)
        offset += particles[i].numberOfData;
    }
    auto temp = new T[numRates * launchParams.numElements];
    resultBuffer.download(temp, launchParams.numElements * numRates);
    offset = (offset + dataIdx) * launchParams.numElements;
    std::memcpy(flux, &temp[offset], launchParams.numElements * sizeof(T));
    delete temp;
  }

  void setUseCellData(unsigned numData) { numCellData = numData; }

  void setPeriodicBoundary(const bool periodic) {
    launchParams.periodicBoundary = periodic;
  }

  void freeBuffers() {
    resultBuffer.free();
    hitgroupRecordBuffer.free();
    missRecordBuffer.free();
    raygenRecordBuffer.free();
    dataPerParticleBuffer.free();
    geometry.freeBuffers();
  }

  unsigned int prepareParticlePrograms() {
    createModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();
    createPipelines();
    if (sbts.size() == 0) {
      for (size_t i = 0; i < particles.size(); i++) {
        OptixShaderBindingTable sbt = {};
        sbts.push_back(sbt);
      }
    }
    numRates = 0;
    std::vector<unsigned int> dataPerParticle;
    for (const auto &p : particles) {
      dataPerParticle.push_back(p.numberOfData);
      numRates += p.numberOfData;
    }
    dataPerParticleBuffer.allocUpload(dataPerParticle);
    launchParams.dataPerParticle =
        (unsigned int *)dataPerParticleBuffer.dPointer();
    return numRates;
  }

  void downloadResultsToPointData(viennals::PointData<T> &pointData,
                                  CudaBuffer &valueBuffer,
                                  unsigned int numPoints) {
    T *temp = new T[numPoints * numRates];
    valueBuffer.download(temp, numPoints * numRates);

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].numberOfData; dIdx++) {
        int tmpOffset = offset + dIdx;
        auto name = particles[pIdx].dataLabels[dIdx];

        std::vector<T> *values = pointData.getScalarData(name, true);
        if (values == nullptr) {
          std::vector<T> val(numPoints);
          pointData.insertNextScalarData(std::move(val), name);
          values = pointData.getScalarData(name);
        }
        if (values->size() != numPoints)
          values->resize(numPoints);

        std::memcpy(values->data(), &temp[tmpOffset * numPoints],
                    numPoints * sizeof(T));
      }
      offset += particles[pIdx].numberOfData;
    }

    delete temp;
  }

  void downloadResultsToPointData(viennals::PointData<T> &pointData) {
    unsigned int numPoints = launchParams.numElements;
    T *temp = new T[numPoints * numRates];
    resultBuffer.download(temp, numPoints * numRates);

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].numberOfData; dIdx++) {
        int tmpOffset = offset + dIdx;
        auto name = particles[pIdx].dataLabels[dIdx];

        std::vector<T> *values = pointData.getScalarData(name, true);
        if (values == nullptr) {
          std::vector<T> val(numPoints);
          pointData.insertNextScalarData(std::move(val), name);
          values = pointData.getScalarData(name);
        }
        if (values->size() != numPoints)
          values->resize(numPoints);

        std::memcpy(values->data(), &temp[tmpOffset * numPoints],
                    numPoints * sizeof(T));
      }
      offset += particles[pIdx].numberOfData;
    }

    delete temp;
  }

  CudaBuffer &getData() { return cellDataBuffer; }

  CudaBuffer &getResults() { return resultBuffer; }

  std::size_t getNumberOfRays() const { return numRays; }

  std::vector<Particle<T>> &getParticles() { return particles; }

  unsigned int getNumberOfRates() const { return numRates; }

  unsigned int getNumberOfElements() const { return launchParams.numElements; }

  SmartPointer<viennals::Mesh<float>> getSurfaceMesh() const { return mesh; }

  SmartPointer<KDTree<T, std::array<float, 3>>> getKDTree() const {
    return kdTree;
  }

protected:
  void normalize() {
    T sourceArea =
        (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]) *
        (launchParams.source.maxPoint[1] - launchParams.source.minPoint[1]);
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initiliazed.");
    CUdeviceptr d_data = resultBuffer.dPointer();
    CUdeviceptr d_vertex = geometry.geometryVertexBuffer.dPointer();
    CUdeviceptr d_index = geometry.geometryIndexBuffer.dPointer();
    void *kernel_args[] = {
        &d_data,     &d_vertex, &d_index, &launchParams.numElements,
        &sourceArea, &numRays,  &numRates};

    LaunchKernel::launch(normModuleName, normKernelName, kernel_args, context);
  }

  void initRayTracer() {

    geometry.optixContext = optixContext;

    launchParamsBuffer.alloc(sizeof(launchParams));
    normKernelName.push_back(NumericType);
    translateFromPointDataKernelName.push_back(NumericType);
    translateToPointDataKernelName.push_back(NumericType);
  }

  /// creates the module that contains all the programs we are going to use. We
  /// use a single module from a single .cu file, using a single embedded ptx
  /// string
  void createModule() {
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        globalParamsName.c_str();

    pipelineLinkOptions.maxTraceDepth = 2;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    size_t inputSize = 0;
    auto pipelineInput =
        getInputData((pipelinePath + pipelineName).c_str(), inputSize);

    OPTIX_CHECK(optixModuleCreate(optixContext, &moduleCompileOptions,
                                  &pipelineCompileOptions, pipelineInput,
                                  inputSize, log, &sizeof_log, &module));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the raygen program
  void createRaygenPrograms() {
    raygenPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__raygen__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module = module;
      pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &raygenPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the miss program
  void createMissPrograms() {
    missPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__miss__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      pgDesc.miss.module = module;
      pgDesc.miss.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &missPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the hitgroup program
  void createHitgroupPrograms() {
    hitgroupPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__closesthit__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      pgDesc.hitgroup.moduleCH = module;
      pgDesc.hitgroup.entryFunctionNameCH = entryFunctionName.c_str();

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &hitgroupPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// assembles the full pipeline of all programs
  void createPipelines() {
    pipelines.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::vector<OptixProgramGroup> programGroups;
      programGroups.push_back(raygenPGs[i]);
      programGroups.push_back(missPGs[i]);
      programGroups.push_back(hitgroupPGs[i]);

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixPipelineCreate(
          optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
          programGroups.data(), (int)programGroups.size(), log, &sizeof_log,
          &pipelines[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
    // probably not needed in current implementation but maybe something to
    // think about in future OPTIX_CHECK(optixPipelineSetStackSize(
    //     pipeline,
    //     2 * 1024, // The direct stack size requirement for direct callables
    //               // invoked from IS or AH.
    //     2 * 1024, // The direct stack size requirement for direct callables
    //               // invoked from RG, MS, or CH.
    //     2 * 1024, // The continuation stack requirement.
    //     1         // The maximum depth of a traversable graph passed to
    //     trace.
    //     ));
  }

  /// constructs the shader binding table
  void buildSBT(const size_t i) {
    // build raygen record
    RaygenRecord raygenRecord;
    optixSbtRecordPackHeader(raygenPGs[i], &raygenRecord);
    raygenRecord.data = nullptr;
    raygenRecordBuffer.allocUploadSingle(raygenRecord);
    sbts[i].raygenRecord = raygenRecordBuffer.dPointer();

    // build miss record
    MissRecord missRecord;
    optixSbtRecordPackHeader(missPGs[i], &missRecord);
    missRecord.data = nullptr;
    missRecordBuffer.allocUploadSingle(missRecord);
    sbts[i].missRecordBase = missRecordBuffer.dPointer();
    sbts[i].missRecordStrideInBytes = sizeof(MissRecord);
    sbts[i].missRecordCount = 1;

    // build hitgroup records
    std::vector<HitgroupRecord> hitgroupRecords;

    // geometry hitgroup
    HitgroupRecord geometryHitgroupRecord;
    optixSbtRecordPackHeader(hitgroupPGs[i], &geometryHitgroupRecord);
    geometryHitgroupRecord.data.vertex =
        (Vec3Df *)geometry.geometryVertexBuffer.dPointer();
    geometryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)geometry.geometryIndexBuffer.dPointer();
    geometryHitgroupRecord.data.isBoundary = false;
    geometryHitgroupRecord.data.cellData = (void *)cellDataBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecord boundaryHitgroupRecord;
    optixSbtRecordPackHeader(hitgroupPGs[i], &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.vertex =
        (Vec3Df *)geometry.boundaryVertexBuffer.dPointer();
    boundaryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)geometry.boundaryIndexBuffer.dPointer();
    boundaryHitgroupRecord.data.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbts[i].hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbts[i].hitgroupRecordCount = 2;
  }

protected:
  // context for cuda kernels
  Context context;
  std::string pipelineName;
  std::string pipelinePath = VIENNAPS_KERNELS_PATH;

  // geometry
  psDomainType domain{nullptr};
  SmartPointer<KDTree<T, std::array<float, 3>>> kdTree{nullptr};
  SmartPointer<viennals::Mesh<float>> mesh{nullptr};

  Geometry<T, D> geometry;

  // particles
  std::vector<Particle<T>> particles;
  CudaBuffer dataPerParticleBuffer;
  unsigned int numRates = 0;

  // sbt data
  CudaBuffer cellDataBuffer;

  // cuda and optix stuff

  std::vector<OptixPipeline> pipelines;
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions pipelineLinkOptions = {};

  OptixModule module;
  OptixModuleCompileOptions moduleCompileOptions = {};

  // program groups, and the SBT built around
  std::vector<OptixProgramGroup> raygenPGs;
  CudaBuffer raygenRecordBuffer;
  std::vector<OptixProgramGroup> missPGs;
  CudaBuffer missRecordBuffer;
  std::vector<OptixProgramGroup> hitgroupPGs;
  CudaBuffer hitgroupRecordBuffer;
  std::vector<OptixShaderBindingTable> sbts;

  // launch parameters, on the host, constant for all particles
  LaunchParams<T> launchParams;
  CudaBuffer launchParamsBuffer;

  // results Buffer
  CudaBuffer resultBuffer;

  bool geometryValid = false;
  bool useRandomSeed = false;
  unsigned numCellData = 0;
  int numberOfRaysPerPoint = 3000;
  int numberOfRaysFixed = 0;
  int runNumber = 0;

  size_t numRays = 0;
  std::string globalParamsName = "params";

  const std::string normModuleName = "normKernels.ptx";
  std::string normKernelName = "normalize_surface_";

  const std::string translateModuleName = "translateKernels.ptx";
  std::string translateToPointDataKernelName = "translate_to_point_cloud_mesh_";
  std::string translateFromPointDataKernelName =
      "translate_from_point_cloud_mesh_";

  static constexpr bool useFloat = std::is_same_v<T, float>;
  static constexpr char NumericType = useFloat ? 'f' : 'd';
};

} // namespace gpu
} // namespace viennaps