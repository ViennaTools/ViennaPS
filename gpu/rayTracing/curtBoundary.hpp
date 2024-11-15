#pragma once

#include <curtPerRayData.hpp>
#include <curtSBTRecords.hpp>
#include <curtUtilities.hpp>

#include <vcVectorUtil.hpp>

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
__device__ __inline__ viennacore::Vec3Df
computeNormal(const viennaps::gpu::HitSBTData *sbt, const unsigned int primID) {
  using namespace viennacore;
  const Vec3D<unsigned> index = sbt->index[primID];
  const Vec3Df &A = sbt->vertex[index[0]];
  const Vec3Df &B = sbt->vertex[index[1]];
  const Vec3Df &C = sbt->vertex[index[2]];
  return Normalize(CrossProduct(B - A, C - A));
}

__device__ __inline__ void reflectFromBoundary(viennaps::gpu::PerRayData *prd) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->pos = prd->pos + prd->dir * optixGetRayTmax();

  if (primID < 4) {
    prd->dir[0] -= 2 * prd->dir[0];
  } else {
    prd->dir[1] -= 2 * prd->dir[1];
  }
}

__device__ __inline__ void
applyPeriodicBoundary(viennaps::gpu::PerRayData *prd,
                      const viennaps::gpu::HitSBTData *hsd) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->pos = prd->pos + optixGetRayTmax() * prd->dir;

  // const uint3 idx = optixGetLaunchIndex();
  // const uint3 dims = optixGetLaunchDimensions();
  // const unsigned int linearLaunchIndex =
  //     idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // printf("Ray %u Hit %u at [%f, %f, %f].\n", linearLaunchIndex, primID,
  //        prd->pos[0], prd->pos[1], prd->pos[2]);

  if (primID == 0 || primID == 1) // x min
  {
    prd->pos[0] = hsd->vertex[hsd->index[2][0]][0]; // set to x max
  } else if (primID == 2 || primID == 3)            // x max
  {
    prd->pos[0] = hsd->vertex[hsd->index[0][0]][0]; // set to x min
  } else if (primID == 4 || primID == 5)            // y min
  {
    prd->pos[1] = hsd->vertex[hsd->index[6][0]][1]; // set to y max
  } else if (primID == 6 || primID == 7)            // y max
  {
    prd->pos[1] = hsd->vertex[hsd->index[4][0]][1]; // set to y min
  }
  // printf("Ray %u Now at [%f, %f, %f].\n", linearLaunchIndex, prd->pos[0],
  //        prd->pos[1], prd->pos[2]);
}
#endif
