#pragma once

#include <curtLaunchParams.hpp>
#include <curtMesh.hpp>

#include <psKDTree.hpp>
#include <psSmartPointer.hpp>

#include <utCudaBuffer.hpp>
#include <utLog.hpp>

#include <culsToSurfaceMesh.hpp>

#include <lsDomain.hpp>

template <typename T, int D> struct curtGeometry {
  // geometry
  utCudaBuffer geometryVertexBuffer;
  utCudaBuffer geometryIndexBuffer;

  // boundary
  utCudaBuffer boundaryVertexBuffer;
  utCudaBuffer boundaryIndexBuffer;

  // buffer that keeps the (final, compacted) accel structure
  utCudaBuffer asBuffer;
  psSmartPointer<psKDTree<T, std::array<T, 3>>> kdTree = nullptr;

  OptixDeviceContext optixContext;

  /// build acceleration structure from level set domain
  template <class LaunchParams>
  void buildAccelFromDomain(psSmartPointer<lsDomain<T, D>> domain,
                            LaunchParams &launchParams) {
    if (domain == nullptr) {
      utLog::getInstance()
          .addError("No level sets passed to curtGeometry.")
          .print();
    }

    auto mesh = psSmartPointer<lsMesh<float>>::New();
    // assert(kdTree);
    // culsToSurfaceMesh<float>(domain, mesh, kdTree).apply();
    culsToSurfaceMesh<float>(domain, mesh).apply();

    const auto gridDelta = domain->getGrid().getGridDelta();
    launchParams.source.gridDelta = gridDelta;
    launchParams.source.minPoint = mesh->minimumExtent;
    launchParams.source.maxPoint = mesh->maximumExtent;
    launchParams.source.planeHeight = mesh->maximumExtent[2] + gridDelta;
    launchParams.numElements = mesh->triangles.size();

    std::vector<OptixBuildInput> triangleInput(2);
    std::vector<uint32_t> triangleInputFlags(2);

    // ------------------- geometry input -------------------
    // upload the model to the device: the builder
    geometryVertexBuffer.alloc_and_upload(mesh->nodes);
    geometryIndexBuffer.alloc_and_upload(mesh->triangles);

    // triangle inputs
    triangleInput[0] = {};
    triangleInput[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_geoVertices = geometryVertexBuffer.d_pointer();
    CUdeviceptr d_geoIndices = geometryIndexBuffer.d_pointer();

    triangleInput[0].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[0].triangleArray.vertexStrideInBytes =
        sizeof(std::array<float, D>);
    triangleInput[0].triangleArray.numVertices =
        (unsigned int)mesh->nodes.size();
    triangleInput[0].triangleArray.vertexBuffers = &d_geoVertices;

    triangleInput[0].triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[0].triangleArray.indexStrideInBytes =
        sizeof(std::array<unsigned, 3>);
    triangleInput[0].triangleArray.numIndexTriplets =
        (unsigned int)mesh->triangles.size();
    triangleInput[0].triangleArray.indexBuffer = d_geoIndices;

    // one SBT entry, and no per-primitive materials:
    triangleInputFlags[0] = 0;
    triangleInput[0].triangleArray.flags = &triangleInputFlags[0];
    triangleInput[0].triangleArray.numSbtRecords = 1;
    triangleInput[0].triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput[0].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput[0].triangleArray.sbtIndexOffsetStrideInBytes = 0;

    // ------------------------- boundary input -------------------------
    TriangleMesh boundaryMesh = makeBoundary(mesh, gridDelta);
    // upload the model to the device: the builder
    boundaryVertexBuffer.alloc_and_upload(boundaryMesh.vertex);
    boundaryIndexBuffer.alloc_and_upload(boundaryMesh.index);

    // triangle inputs
    triangleInput[1] = {};
    triangleInput[1].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_boundVertices = boundaryVertexBuffer.d_pointer();
    CUdeviceptr d_boundIndices = boundaryIndexBuffer.d_pointer();

    triangleInput[1].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[1].triangleArray.vertexStrideInBytes = sizeof(gdt::vec3f);
    triangleInput[1].triangleArray.numVertices =
        (int)boundaryMesh.vertex.size();
    triangleInput[1].triangleArray.vertexBuffers = &d_boundVertices;

    triangleInput[1].triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[1].triangleArray.indexStrideInBytes = sizeof(gdt::vec3i);
    triangleInput[1].triangleArray.numIndexTriplets =
        (int)boundaryMesh.index.size();
    triangleInput[1].triangleArray.indexBuffer = d_boundIndices;

    // one SBT entry, and no per-primitive materials:
    triangleInputFlags[1] = 0;
    triangleInput[1].triangleArray.flags = &triangleInputFlags[1];
    triangleInput[1].triangleArray.numSbtRecords = 1;
    triangleInput[1].triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput[1].triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput[1].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    // ------------------------------------------------------

    OptixTraversableHandle asHandle{0};

    // BLAS setup
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixAccelComputeMemoryUsage(optixContext, &accelOptions,
                                 triangleInput.data(),
                                 2, // num_build_inputs
                                 &blasBufferSizes);

    // prepare compaction
    utCudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // execute build
    utCudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    utCudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixAccelBuild(optixContext, 0, &accelOptions, triangleInput.data(), 2,
                    tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
                    outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
                    &asHandle, &emitDesc, 1);
    cudaDeviceSynchronize();

    // perform compaction
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    optixAccelCompact(optixContext, 0, asHandle, asBuffer.d_pointer(),
                      asBuffer.sizeInBytes, &asHandle);
    cudaDeviceSynchronize();

    // clean up
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    launchParams.traversable = asHandle;
  }

  static TriangleMesh makeBoundary(psSmartPointer<lsMesh<float>> passedMesh,
                                   const float gridDelta) {
    TriangleMesh boundaryMesh;

    gdt::vec3f bbMin = passedMesh->minimumExtent;
    gdt::vec3f bbMax = passedMesh->maximumExtent;
    // adjust bounding box to include source plane
    bbMax.z += gridDelta;

    boundaryMesh.index.reserve(8);
    boundaryMesh.vertex.reserve(8);

    // bottom
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMin.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMin.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMax.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMax.y, bbMin.z));
    // top
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMin.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMin.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMax.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMax.y, bbMax.z));

    // x min max
    boundaryMesh.index.push_back(gdt::vec3i(0, 3, 7));
    boundaryMesh.index.push_back(gdt::vec3i(0, 7, 4));
    boundaryMesh.index.push_back(gdt::vec3i(6, 2, 1));
    boundaryMesh.index.push_back(gdt::vec3i(6, 1, 5));
    // y min max
    boundaryMesh.index.push_back(gdt::vec3i(0, 4, 5));
    boundaryMesh.index.push_back(gdt::vec3i(0, 5, 1));
    boundaryMesh.index.push_back(gdt::vec3i(6, 7, 3));
    boundaryMesh.index.push_back(gdt::vec3i(6, 3, 2));

    boundaryMesh.minCoords = bbMin;
    boundaryMesh.maxCoords = bbMax;

    return boundaryMesh;
  }

  static TriangleMesh makeBoundary(const TriangleMesh &model) {
    TriangleMesh boundaryMesh;

    gdt::vec3f bbMin = model.minCoords;
    gdt::vec3f bbMax = model.maxCoords;
    // adjust bounding box to include source plane
    bbMax.z += model.gridDelta;

    boundaryMesh.index.reserve(8);
    boundaryMesh.vertex.reserve(8);

    // bottom
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMin.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMin.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMax.y, bbMin.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMax.y, bbMin.z));
    // top
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMin.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMin.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMax.x, bbMax.y, bbMax.z));
    boundaryMesh.vertex.push_back(gdt::vec3f(bbMin.x, bbMax.y, bbMax.z));

    // x min max
    boundaryMesh.index.push_back(gdt::vec3i(0, 3, 7));
    boundaryMesh.index.push_back(gdt::vec3i(0, 7, 4));
    boundaryMesh.index.push_back(gdt::vec3i(6, 2, 1));
    boundaryMesh.index.push_back(gdt::vec3i(6, 1, 5));
    // y min max
    boundaryMesh.index.push_back(gdt::vec3i(0, 4, 5));
    boundaryMesh.index.push_back(gdt::vec3i(0, 5, 1));
    boundaryMesh.index.push_back(gdt::vec3i(6, 7, 3));
    boundaryMesh.index.push_back(gdt::vec3i(6, 3, 2));

    boundaryMesh.minCoords = bbMin;
    boundaryMesh.maxCoords = bbMax;

    return boundaryMesh;
  }

  void freeBuffers() {
    geometryIndexBuffer.free();
    geometryVertexBuffer.free();
    boundaryIndexBuffer.free();
    boundaryVertexBuffer.free();
    asBuffer.free();
  }
};
