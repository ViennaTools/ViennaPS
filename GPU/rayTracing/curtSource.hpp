#pragma once

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>
#include <curtUtilities.hpp>

#ifdef __CUDACC__
template <typename T>
__device__ void initializeDirection(PerRayData<T> *prd, const T power) {
  // source direction
  auto r1 = getNextRand(&prd->RNGstate);
  auto r2 = getNextRand(&prd->RNGstate);
  const T ee = 2.f / (power + 1.f);
  const T tt = pow(r2, ee);
  prd->dir.z = -1. * sqrtf(tt);
  prd->dir.x = cosf(2 * PI_F * r1) * sqrtf(1 - tt);
  prd->dir.y = sinf(2 * PI_F * r1) * sqrtf(1 - tt);
  gdt::normalize(prd->dir);
}

template <typename T>
__device__ void initializeRayOnGrid(PerRayData<T> *prd,
                                    curtLaunchParams<T> *launchParams,
                                    const T power, uint3 launchIdx) {
  // initial position on source plane
  gdt::vec_t<T, 2> sourceVoxel(launchIdx.x * launchParams->source.gridDelta +
                                   launchParams->source.minPoint.x,
                               launchIdx.y * launchParams->source.gridDelta +
                                   launchParams->source.minPoint.y);

  const T subGridDelta =
      launchParams->source.gridDelta / (T)(launchParams->voxelDim + 1);
  const T xOffset = (launchIdx.z % launchParams->voxelDim + 1) * subGridDelta;
  const T yOffset = (launchIdx.z / launchParams->voxelDim + 1) * subGridDelta;
  prd->pos.x = sourceVoxel.x + xOffset;
  prd->pos.y = sourceVoxel.y + yOffset;
  prd->pos.z = launchParams->source.planeHeight;

  initializeDirection<T>(prd, power);
}

template <typename T, typename LaunchParams>
__device__ void initializeRayRandom(PerRayData<T> *prd,
                                    LaunchParams *launchParams, const T power,
                                    uint3 launchIdx) {
  auto rx = getNextRand(&prd->RNGstate);
  auto ry = getNextRand(&prd->RNGstate);
  prd->pos.x =
      launchParams->source.minPoint.x +
      rx * (launchParams->source.maxPoint.x - launchParams->source.minPoint.x);
  prd->pos.y =
      launchParams->source.minPoint.y +
      ry * (launchParams->source.maxPoint.y - launchParams->source.minPoint.y);
  prd->pos.z = launchParams->source.planeHeight;

  initializeDirection(prd, power);
}
#endif