#pragma once

#include <curtPerRayData.hpp>
#include <curtSBTRecords.hpp>
#include <curtUtilities.hpp>

#include <utGDT.hpp>

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
__device__ __inline__ gdt::vec3f computeNormal(const HitSBTData *sbt,
                                               const unsigned int primID) {
  const gdt::vec3i index = sbt->index[primID];
  const gdt::vec3f &A = sbt->vertex[index.x];
  const gdt::vec3f &B = sbt->vertex[index.y];
  const gdt::vec3f &C = sbt->vertex[index.z];
  return gdt::normalize(gdt::cross(B - A, C - A));
}

__device__ __inline__ void reflectFromBoundary(PerRayData *prd) {
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->pos = prd->pos + optixGetRayTmax() * prd->dir;

  if (primID < 4) {
    prd->dir.x -= 2 * prd->dir.x;
  } else {
    prd->dir.y -= 2 * prd->dir.y;
  }
}

__device__ __inline__ void applyPeriodicBoundary(PerRayData *prd,
                                                 const HitSBTData *hsd) {
  const unsigned int primID = optixGetPrimitiveIndex();

  // printf("Hit %u at [%f, %f, %f].", primID, prd->pos.x, prd->pos.y,
  // prd->pos.z);
  if (primID == 0 || primID == 1) {
    prd->pos.x = hsd->vertex[hsd->index[0].x].x; // set to x min
  } else if (primID == 2 || primID == 3) {
    prd->pos.x = hsd->vertex[hsd->index[2].x].x; // set to x max
  } else if (primID == 4 || primID == 5) {
    prd->pos.y = hsd->vertex[hsd->index[4].x].y; // set to y min
  } else if (primID == 6 || primID == 7) {
    prd->pos.y = hsd->vertex[hsd->index[6].x].y; // set to y max
  }
  // printf("  Now at [%f, %f, %f].\n", prd->pos.x, prd->pos.y, prd->pos.z);
}
#endif
