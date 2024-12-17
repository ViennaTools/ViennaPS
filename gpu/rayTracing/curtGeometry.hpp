#pragma once

#include <curtLaunchParams.hpp>
#include <curtMesh.hpp>

#include <vcKDTree.hpp>

#include <utCudaBuffer.hpp>

#include <lsToSurfaceMeshRefined.hpp>

#include <psDomain.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename T, int D> struct Geometry {
  // geometry
  CudaBuffer geometryVertexBuffer;
  CudaBuffer geometryIndexBuffer;

  // boundary
  CudaBuffer boundaryVertexBuffer;
  CudaBuffer boundaryIndexBuffer;

  // buffer that keeps the (final, compacted) accel structure
  CudaBuffer asBuffer;
  OptixDeviceContext optixContext;

  /// build acceleration structure from level set domain
  template <class LaunchParams>
  void buildAccelFromDomain(SmartPointer<::viennaps::Domain<T, D>> domain,
                            LaunchParams &launchParams,
                            SmartPointer<::viennals::Mesh<float>> mesh,
                            SmartPointer<KDTree<T, Vec3D<T>>> kdTree) {
    if (domain == nullptr) {
      Logger::getInstance()
          .addError("No level sets passed to Geometry.")
          .print();
    }

    if (kdTree == nullptr) {
      Logger::getInstance().addWarning("No KDTree passed to Geometry.").print();
    }

    // create surface mesh
    viennals::ToSurfaceMeshRefined<T, float, D>(domain->getLevelSets().back(),
                                                mesh, kdTree)
        .apply();

    const auto gridDelta = domain->getGrid().getGridDelta();
    launchParams.source.gridDelta = gridDelta;
    launchParams.source.minPoint[0] = mesh->minimumExtent[0];
    launchParams.source.minPoint[1] = mesh->minimumExtent[1];
    launchParams.source.maxPoint[0] = mesh->maximumExtent[0];
    launchParams.source.maxPoint[1] = mesh->maximumExtent[1];
    launchParams.source.planeHeight = mesh->maximumExtent[2] + gridDelta;
    launchParams.numElements = mesh->triangles.size();

    std::array<OptixBuildInput, 2> triangleInput;
    std::array<uint32_t, 2> triangleInputFlags;

    // ------------------- geometry input -------------------
    // upload the model to the device: the builder
    geometryVertexBuffer.allocUpload(mesh->nodes);
    geometryIndexBuffer.allocUpload(mesh->triangles);

    // triangle inputs
    triangleInput[0] = {};
    triangleInput[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_geoVertices = geometryVertexBuffer.dPointer();
    CUdeviceptr d_geoIndices = geometryIndexBuffer.dPointer();

    triangleInput[0].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[0].triangleArray.vertexStrideInBytes = sizeof(Vec3Df);
    triangleInput[0].triangleArray.numVertices =
        (unsigned int)mesh->nodes.size();
    triangleInput[0].triangleArray.vertexBuffers = &d_geoVertices;

    triangleInput[0].triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[0].triangleArray.indexStrideInBytes = sizeof(Vec3D<unsigned>);
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
    boundaryVertexBuffer.allocUpload(boundaryMesh.vertex);
    boundaryIndexBuffer.allocUpload(boundaryMesh.index);

    // triangle inputs
    triangleInput[1] = {};
    triangleInput[1].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_boundVertices = boundaryVertexBuffer.dPointer();
    CUdeviceptr d_boundIndices = boundaryIndexBuffer.dPointer();

    triangleInput[1].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[1].triangleArray.vertexStrideInBytes = sizeof(Vec3Df);
    triangleInput[1].triangleArray.numVertices =
        (int)boundaryMesh.vertex.size();
    triangleInput[1].triangleArray.vertexBuffers = &d_boundVertices;

    triangleInput[1].triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[1].triangleArray.indexStrideInBytes = sizeof(Vec3D<unsigned>);
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
    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.dPointer();

    // execute build
    CudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixAccelBuild(optixContext, 0, &accelOptions, triangleInput.data(), 2,
                    tempBuffer.dPointer(), tempBuffer.sizeInBytes,
                    outputBuffer.dPointer(), outputBuffer.sizeInBytes,
                    &asHandle, &emitDesc, 1);
    cudaDeviceSynchronize();

    // perform compaction
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    optixAccelCompact(optixContext, 0, asHandle, asBuffer.dPointer(),
                      asBuffer.sizeInBytes, &asHandle);
    cudaDeviceSynchronize();

    // clean up
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    launchParams.traversable = asHandle;
  }

  static TriangleMesh
  makeBoundary(SmartPointer<viennals::Mesh<float>> passedMesh,
               const float gridDelta) {
    TriangleMesh boundaryMesh;

    Vec3Df bbMin = passedMesh->minimumExtent;
    Vec3Df bbMax = passedMesh->maximumExtent;
    // adjust bounding box to include source plane
    bbMax[2] += gridDelta;

    boundaryMesh.index.reserve(8);
    boundaryMesh.vertex.reserve(8);

    // bottom
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMin[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMin[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMax[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMax[1], bbMin[2]});
    // top
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMin[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMin[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMax[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMax[1], bbMax[2]});

    // x min max
    boundaryMesh.index.push_back(Vec3D<int>{0, 3, 7}); // 0
    boundaryMesh.index.push_back(Vec3D<int>{0, 7, 4}); // 1
    boundaryMesh.index.push_back(Vec3D<int>{6, 2, 1}); // 2
    boundaryMesh.index.push_back(Vec3D<int>{6, 1, 5}); // 3
    // y min max
    boundaryMesh.index.push_back(Vec3D<int>{0, 4, 5}); // 4
    boundaryMesh.index.push_back(Vec3D<int>{0, 5, 1}); // 5
    boundaryMesh.index.push_back(Vec3D<int>{6, 7, 3}); // 6
    boundaryMesh.index.push_back(Vec3D<int>{6, 3, 2}); // 7

    boundaryMesh.minCoords = bbMin;
    boundaryMesh.maxCoords = bbMax;

    return boundaryMesh;
  }

  static TriangleMesh makeBoundary(const TriangleMesh &model) {
    TriangleMesh boundaryMesh;

    Vec3Df bbMin = model.minCoords;
    Vec3Df bbMax = model.maxCoords;
    // adjust bounding box to include source plane
    bbMax[2] += model.gridDelta;

    boundaryMesh.index.reserve(8);
    boundaryMesh.vertex.reserve(8);

    // bottom
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMin[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMin[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMax[1], bbMin[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMax[1], bbMin[2]});
    // top
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMin[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMin[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMax[0], bbMax[1], bbMax[2]});
    boundaryMesh.vertex.push_back(Vec3Df{bbMin[0], bbMax[1], bbMax[2]});

    // x min max
    boundaryMesh.index.push_back(Vec3D<int>{0, 3, 7});
    boundaryMesh.index.push_back(Vec3D<int>{0, 7, 4});
    boundaryMesh.index.push_back(Vec3D<int>{6, 2, 1});
    boundaryMesh.index.push_back(Vec3D<int>{6, 1, 5});
    // y min max
    boundaryMesh.index.push_back(Vec3D<int>{0, 4, 5});
    boundaryMesh.index.push_back(Vec3D<int>{0, 5, 1});
    boundaryMesh.index.push_back(Vec3D<int>{6, 7, 3});
    boundaryMesh.index.push_back(Vec3D<int>{6, 3, 2});

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

} // namespace gpu
} // namespace viennaps