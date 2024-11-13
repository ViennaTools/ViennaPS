#pragma once

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtUtilities.hpp>

#ifdef __CUDACC__
__device__ void initializeDirection(viennaps::gpu::PerRayData *prd, const float power) {
  // source direction
  auto r1 = getNextRand(&prd->RNGstate);
  auto r2 = getNextRand(&prd->RNGstate);
  const float ee = 2.f / (power + 1.f);
  const float tt = powf(r2, ee);
  prd->dir[0] = cosf(2 * PI_F * r1) * sqrtf(1 - tt);
  prd->dir[1] = sinf(2 * PI_F * r1) * sqrtf(1 - tt);
  prd->dir[2] = -1.f * sqrtf(tt);
  viennacore::Normalize(prd->dir);
}

// ** DEPRECATED ** //
// template <typename T>
// __device__ void initializeRayOnGrid(PerRayData *prd,
//                                     curtLaunchParams<T> *launchParams,
//                                     const T power, uint3 launchIdx) {
//   // initial position on source plane
//   gdt::vec_t<T, 2> sourceVoxel(launchIdx[0] * launchParams->source.gridDelta +
//                                    launchParams->source.minPoint[0],
//                                launchIdx[1] * launchParams->source.gridDelta +
//                                    launchParams->source.minPoint[1]);

//   const T subGridDelta =
//       launchParams->source.gridDelta / (T)(launchParams->voxelDim + 1);
//   const T xOffset = (launchIdx[2] % launchParams->voxelDim + 1) *
//   subGridDelta; const T yOffset = (launchIdx[2] / launchParams->voxelDim + 1)
//   * subGridDelta; prd->pos[0] = sourceVoxel[0] + xOffset; prd->pos[1] =
//   sourceVoxel[1] + yOffset; prd->pos[2] = launchParams->source.planeHeight;

//   initializeDirection<T>(prd, power);
// }

template <typename LaunchParams>
__device__ void initializeRayRandom(viennaps::gpu::PerRayData *prd, LaunchParams *launchParams,
                                    const float power, uint3 launchIdx) {
  auto rx = getNextRand(&prd->RNGstate);
  auto ry = getNextRand(&prd->RNGstate);
  prd->pos[0] =
      launchParams->source.minPoint[0] +
      rx * (launchParams->source.maxPoint[0] - launchParams->source.minPoint[0]);
  prd->pos[1] =
      launchParams->source.minPoint[1] +
      ry * (launchParams->source.maxPoint[1] - launchParams->source.minPoint[1]);
  prd->pos[2] = launchParams->source.planeHeight;

  initializeDirection(prd, power);
}
#endif