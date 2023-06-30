#pragma once

#include "context.hpp"
#include <cuda.h>
#include <optix_stubs.h>

#include <cstring>

#include <lsDomain.hpp>
#include <lsPointData.hpp>

#include <psKDTree.hpp>

#include <curtBoundary.hpp>
#include <curtChecks.hpp>
#include <curtGeometry.hpp>
#include <curtLaunchParams.hpp>
#include <curtParticle.hpp>
#include <curtSBTRecords.hpp>
#include <curtUtilities.hpp>

#include <utCudaBuffer.hpp>
#include <utGDT.hpp>
#include <utLaunchKernel.hpp>

template <class T, int D> class curtTracer {
public:
  /// constructor - performs all setup, including initializing
  /// optix, creates module, pipeline, programs, SBT, etc.
  curtTracer(pscuContext passedContext,
             psSmartPointer<lsDomain<T, D>> passedDomain)
      : context(passedContext), domain(passedDomain) {
    initRayTracer();
  }

  curtTracer(pscuContext passedContext) : context(passedContext) {
    initRayTracer();
  }

  void setKdTree(psSmartPointer<psKDTree<T, std::array<T, 3>>> passedKdTree) {
    geometry.kdTree = passedKdTree;
  }

  void setPipeline(char embeddedPtxCode[]) { ptxCode = embeddedPtxCode; }

  void setLevelSet(psSmartPointer<lsDomain<T, D>> passedDomain) {
    domain = passedDomain;
    // double start = gdt::getCurrentTime();
    geometry.buildAccelFromDomain(domain, launchParams);
    // geoBuildTime += gdt::getCurrentTime() - start;
    geometryValid = true;
  }

  void invalidateGeometry() { geometryValid = false; }

  void insertNextParticle(const curtParticle<T> &particle) {
    particles.push_back(particle);
  }

  void apply() {
    if (!geometryValid) {
      // double start = gdt::getCurrentTime();
      geometry.buildAccelFromDomain(domain, launchParams);
      // geoBuildTime += gdt::getCurrentTime() - start;
      geometryValid = true;
    }

    if (numCellData != 0 && cellDataBuffer.sizeInBytes == 0) {
      cellDataBuffer.alloc_and_init(numCellData * launchParams.numElements,
                                    T(0));
    }
    assert(cellDataBuffer.sizeInBytes / sizeof(T) ==
           numCellData * launchParams.numElements);

    // resize our cuda result buffer
    resultBuffer.alloc_and_init(launchParams.numElements * numRates, T(0));
    launchParams.resultBuffer = (T *)resultBuffer.d_pointer();

    if (useRandomSeed) {
      std::random_device rd;
      std::uniform_int_distribution<unsigned int> gen;
      launchParams.seed = gen(rd);
    }

    if (numberOfRaysPerPoint == 1) {
      launchParams.voxelDim = numberOfRaysPerPoint;
    } else {
      int i = 1, result = 1;
      while (result <= numberOfRaysPerPoint) {
        i++;
        result = i * i;
      }
      launchParams.voxelDim = i - 1;
    }

    int numPoints_x = static_cast<int>(0.5 + (launchParams.source.maxPoint.x -
                                              launchParams.source.minPoint.x) /
                                                 launchParams.source.gridDelta);
    int numPoints_y = static_cast<int>(0.5 + (launchParams.source.maxPoint.y -
                                              launchParams.source.minPoint.y) /
                                                 launchParams.source.gridDelta);

    if (numberOfRaysFixed > 0) {
      numPoints_x = 1;
      numPoints_y = 1;
      numberOfRaysPerPoint = numberOfRaysFixed;
    }

    if (numPoints_x * numPoints_y * numberOfRaysPerPoint > (1 << 29)) {
      utLog::getInstance().addError("Too many rays for single launch.").print();
    }

    numRays = numPoints_x * numPoints_y * numberOfRaysPerPoint;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < particles.size(); i++) {

      launchParams.cosineExponent = particles[i].cosineExponent;
      launchParams.sticking = particles[i].sticking;
      launchParams.meanIonEnergy = particles[i].meanIonEnergy;
      launchParams.ionRF = particles[i].ionRF;
      launchParams.A_O = particles[i].A_O;
      launchParamsBuffer.upload(&launchParams, 1);

      CUstream stream;
      CUDA_CHECK(StreamCreate(&stream));
      buildSBT(i);
      OPTIX_CHECK(optixLaunch(pipelines[i], stream,
                              /*! parameters and SBT */
                              launchParamsBuffer.d_pointer(),
                              launchParamsBuffer.sizeInBytes, &sbts[i],
                              /*! dimensions of the launch: */
                              numPoints_x, numPoints_y, numberOfRaysPerPoint));
    }

    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // T *temp = new T[launchParams.numElements];
    // resultBuffer.download(temp, launchParams.numElements);
    // for (int i = 0; i < launchParams.numElements; i++) {
    //   std::cout << temp[i] << std::endl;
    // }
    // delete temp;
    // std::cout << "Time: " << diff.count() * 100 << " ms\n";
    std::cout << gdt::prettyDouble(numRays * particles.size()) << std::endl;
    std::cout << numRays << std::endl;

    // sync - maybe remove in future
    cudaDeviceSynchronize();
    normalize();
  }

  void translateToPointData(psSmartPointer<lsMesh<T>> mesh,
                            utCudaBuffer &pointDataBuffer, T radius = 0.,
                            const bool download = false) {
    // upload oriented pointcloud data to device
    assert(mesh->nodes.size() != 0 &&
           "Passing empty mesh in translateToPointValuesSphere.");
    if (radius == 0.)
      radius = launchParams.source.gridDelta;
    size_t numValues = mesh->nodes.size();
    utCudaBuffer pointBuffer;
    pointBuffer.alloc_and_upload(mesh->nodes);
    pointDataBuffer.alloc_and_init(numValues * numRates, T(0));

    CUdeviceptr d_vertex = geometry.geometryVertexBuffer.d_pointer();
    CUdeviceptr d_index = geometry.geometryIndexBuffer.d_pointer();
    CUdeviceptr d_values = resultBuffer.d_pointer();
    CUdeviceptr d_point = pointBuffer.d_pointer();
    CUdeviceptr d_pointValues = pointDataBuffer.d_pointer();

    void *kernel_args[] = {
        &d_vertex,      &d_index, &d_values,  &d_point,
        &d_pointValues, &radius,  &numValues, &launchParams.numElements,
        &numRates};

    utLaunchKernel::launch(translateModuleName, translateToPointDataKernelName,
                           kernel_args, context, sizeof(int));

    if (download) {
      downloadResultsToPointData(mesh->getCellData(), pointDataBuffer,
                                 mesh->nodes.size());
    }

    pointBuffer.free();
  }

  void translateFromPointData(psSmartPointer<lsMesh<T>> mesh,
                              utCudaBuffer &pointDataBuffer, unsigned numData) {
    // upload oriented pointcloud data to device
    size_t numPoints = mesh->nodes.size();
    assert(mesh->nodes.size() > 0);
    assert(pointDataBuffer.sizeInBytes / sizeof(T) / numData == numPoints);
    assert(numData > 0);

    utCudaBuffer pointBuffer;
    pointBuffer.alloc_and_upload(mesh->nodes);

    cellDataBuffer.alloc(launchParams.numElements * numData * sizeof(T));

    CUdeviceptr d_vertex = geometry.geometryVertexBuffer.d_pointer();
    CUdeviceptr d_index = geometry.geometryIndexBuffer.d_pointer();
    CUdeviceptr d_values = cellDataBuffer.d_pointer();
    CUdeviceptr d_point = pointBuffer.d_pointer();
    CUdeviceptr d_pointValues = pointDataBuffer.d_pointer();

    void *kernel_args[] = {&d_vertex,
                           &d_index,
                           &d_values,
                           &d_point,
                           &d_pointValues,
                           &numPoints,
                           &launchParams.numElements,
                           &numData};

    utLaunchKernel::launch(translateModuleName,
                           translateFromPointDataKernelName, kernel_args,
                           context, sizeof(int));

    pointBuffer.free();
  }

  void updateSurface() {
    // double start = gdt::getCurrentTime();
    geometry.buildAccelFromDomain(domain, launchParams);
    // geoBuildTime += gdt::getCurrentTime() - start;
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

  utCudaBuffer &getData() { return cellDataBuffer; }

  utCudaBuffer &getResults() { return resultBuffer; }

  size_t getNumberOfRays() const { return numRays; }

  void freeBuffers() {
    resultBuffer.free();
    hitgroupRecordBuffer.free();
    missRecordBuffer.free();
    raygenRecordBuffer.free();
    dataPerParticleBuffer.free();
    geometry.freeBuffers();
  }

  void prepareParticlePrograms() {
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
    dataPerParticleBuffer.alloc_and_upload(dataPerParticle);
    launchParams.dataPerParticle =
        (unsigned int *)dataPerParticleBuffer.d_pointer();
  }

  void downloadResultsToPointData(lsPointData<T> &pointData,
                                  utCudaBuffer &valueBuffer, size_t numPoints) {
    T *temp = new T[numPoints * numRates];
    valueBuffer.download(temp, numPoints * numRates);

    int offset = 0;
    for (int pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (int dIdx = 0; dIdx < particles[pIdx].numberOfData; dIdx++) {
        int tmpOffset = offset + dIdx;
        auto name = particles[pIdx].dataLabels[dIdx];

        std::vector<T> *values = pointData.getScalarData(name);
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

  std::vector<curtParticle<T>> &getParticles() { return particles; }

  unsigned int getNumberOfRates() const { return numRates; }

  double geoBuildTime = 0;

protected:
  void normalize() {
    T sourceArea =
        (launchParams.source.maxPoint.x - launchParams.source.minPoint.x) *
        (launchParams.source.maxPoint.y - launchParams.source.minPoint.y);
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initiliazed.");
    CUdeviceptr d_data = resultBuffer.d_pointer();
    CUdeviceptr d_vertex = geometry.geometryVertexBuffer.d_pointer();
    CUdeviceptr d_index = geometry.geometryIndexBuffer.d_pointer();
    void *kernel_args[] = {
        &d_data,     &d_vertex, &d_index, &launchParams.numElements,
        &sourceArea, &numRays,  &numRates};

    utLaunchKernel::launch(normModuleName, normKernelName, kernel_args,
                           context);
  }

  void initRayTracer() {
    createContext();

    geometry.optixContext = optixContext;

    launchParamsBuffer.alloc(sizeof(launchParams));
    normKernelName.push_back(NumericType);
    translateFromPointDataKernelName.push_back(NumericType);
    translateToPointDataKernelName.push_back(NumericType);
  }

  static void context_log_cb(unsigned int level, const char *tag,
                             const char *message, void *) {
#ifndef NDEBUG
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
#endif
  }

  /// creates and configures a optix device context (only for the primary GPU
  /// device)
  void createContext() {
    const int deviceID = 0;
    cudaSetDevice(deviceID);

    cudaGetDeviceProperties(&deviceProps, deviceID);

#ifndef NDEBUG
    utLog::getInstance()
        .addDebug("Running on device: " + std::string(deviceProps.name))
        .print();
#endif

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS) {
      utLog::getInstance()
          .addError("Error querying current context: error code " +
                    std::to_string(cuRes))
          .print();
    }

    optixDeviceContextCreate(cudaContext, 0, &optixContext);
    optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4);
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
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optixContext, &moduleCompileOptions, &pipelineCompileOptions,
        ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
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
    raygenRecordBuffer.alloc_and_upload_single(raygenRecord);
    sbts[i].raygenRecord = raygenRecordBuffer.d_pointer();

    // build miss record
    MissRecord missRecord;
    optixSbtRecordPackHeader(missPGs[i], &missRecord);
    missRecord.data = nullptr;
    missRecordBuffer.alloc_and_upload_single(missRecord);
    sbts[i].missRecordBase = missRecordBuffer.d_pointer();
    sbts[i].missRecordStrideInBytes = sizeof(MissRecord);
    sbts[i].missRecordCount = 1;

    // build hitgroup records
    std::vector<HitgroupRecord> hitgroupRecords;

    // geometry hitgroup
    HitgroupRecord geometryHitgroupRecord;
    optixSbtRecordPackHeader(hitgroupPGs[i], &geometryHitgroupRecord);
    geometryHitgroupRecord.data.vertex =
        (gdt::vec3f *)geometry.geometryVertexBuffer.d_pointer();
    geometryHitgroupRecord.data.index =
        (gdt::vec3i *)geometry.geometryIndexBuffer.d_pointer();
    geometryHitgroupRecord.data.isBoundary = false;
    geometryHitgroupRecord.data.cellData = (void *)cellDataBuffer.d_pointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecord boundaryHitgroupRecord;
    optixSbtRecordPackHeader(hitgroupPGs[i], &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.vertex =
        (gdt::vec3f *)geometry.boundaryVertexBuffer.d_pointer();
    boundaryHitgroupRecord.data.index =
        (gdt::vec3i *)geometry.boundaryIndexBuffer.d_pointer();
    boundaryHitgroupRecord.data.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.alloc_and_upload(hitgroupRecords);
    sbts[i].hitgroupRecordBase = hitgroupRecordBuffer.d_pointer();
    sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbts[i].hitgroupRecordCount = 2;
  }

protected:
  // context for cuda kernels
  pscuContext_t *context;
  std::string ptxCode;

  // geometry
  psSmartPointer<lsDomain<T, D>> domain;
  curtGeometry<T, D> geometry;

  // particles
  std::vector<curtParticle<T>> particles;
  utCudaBuffer dataPerParticleBuffer;
  unsigned int numRates = 0;

  // sbt data
  utCudaBuffer cellDataBuffer;

  // cuda and optix stuff
  CUcontext cudaContext;
  cudaDeviceProp deviceProps;
  OptixDeviceContext optixContext;

  std::vector<OptixPipeline> pipelines;
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions pipelineLinkOptions = {};

  OptixModule module;
  OptixModuleCompileOptions moduleCompileOptions = {};

  // program groups, and the SBT built around
  std::vector<OptixProgramGroup> raygenPGs;
  utCudaBuffer raygenRecordBuffer;
  std::vector<OptixProgramGroup> missPGs;
  utCudaBuffer missRecordBuffer;
  std::vector<OptixProgramGroup> hitgroupPGs;
  utCudaBuffer hitgroupRecordBuffer;
  std::vector<OptixShaderBindingTable> sbts;

  // launch parameters, on the host, constant for all particles
  curtLaunchParams<T> launchParams;
  utCudaBuffer launchParamsBuffer;

  // results Buffer
  utCudaBuffer resultBuffer;

  bool geometryValid = false;
  bool useRandomSeed = false;
  unsigned numCellData = 0;
  int numberOfRaysPerPoint = 50 * 50;
  int numberOfRaysFixed = 0;

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