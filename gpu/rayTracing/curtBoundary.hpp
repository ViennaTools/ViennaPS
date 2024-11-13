#pragma once

#include <curtPerRayData.hpp>
#include <curtSBTRecords.hpp>
#include <curtUtilities.hpp>

#include <utGDT.hpp>
#include <vcVectorUtil.hpp>

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
__device__ __inline__ viennacore::Vec3Df
computeNormal(const HitSBTData *sbt, const unsigned int primID) {
  using namespace viennacore;
  const Vec3D<int> index = sbt->index[primID];
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

__device__ __inline__ void applyPeriodicBoundary(viennaps::gpu::PerRayData *prd,
                                                 const HitSBTData *hsd) {
  const unsigned int primID = optixGetPrimitiveIndex();

  // printf("Hit %u at [%f, %f, %f].", primID, prd->pos[0], prd->pos[1],
  // prd->pos[2]);
  if (primID == 0 || primID == 1) {
    prd->pos[0] = hsd->vertex[hsd->index[0][0]][0]; // set to x min
  } else if (primID == 2 || primID == 3) {
    prd->pos[0] = hsd->vertex[hsd->index[2][0]][0]; // set to x max
  } else if (primID == 4 || primID == 5) {
    prd->pos[1] = hsd->vertex[hsd->index[4][0]][1]; // set to y min
  } else if (primID == 6 || primID == 7) {
    prd->pos[1] = hsd->vertex[hsd->index[6][0]][1]; // set to y max
  }
  // printf("  Now at [%f, %f, %f].\n", prd->pos[0], prd->pos[1], prd->pos[2]);
}
#endif
