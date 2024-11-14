#pragma once

#include <curtLaunchParams.hpp>
#include <curtPerRayData.hpp>
#include <curtRNG.hpp>

#include <vcVectorUtil.hpp>

#ifdef __CUDACC__
__device__ std::array<viennacore::Vec3Df, 3>
getOrthonormalBasis(const viennacore::Vec3Df &vec)
{
  std::array<viennacore::Vec3Df, 3> rr;
  rr[0] = vec;

  // Calculate a vector (rr[1]) which is perpendicular to rr[0]
  // https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector#answer-211195
  viennacore::Vec3Df candidate0{rr[0][2], rr[0][2], -(rr[0][0] + rr[0][1])};
  viennacore::Vec3Df candidate1{rr[0][1], -(rr[0][0] + rr[0][2]), rr[0][1]};
  viennacore::Vec3Df candidate2{-(rr[0][1] + rr[0][2]), rr[0][0], rr[0][0]};
  // We choose the candidate which maximizes the sum of its components,
  // because we want to avoid numeric errors and that the result is (0, 0, 0).
  std::array<viennacore::Vec3Df, 3> cc = {candidate0, candidate1, candidate2};
  auto sumFun = [](const viennacore::Vec3Df &oo)
  {
    return oo[0] + oo[1] + oo[2];
  };
  int maxIdx = 0;
  for (size_t idx = 1; idx < cc.size(); ++idx)
  {
    if (sumFun(cc[idx]) > sumFun(cc[maxIdx]))
    {
      maxIdx = idx;
    }
  }
  assert(maxIdx < 3 && "Error in computation of perpendicular vector");
  rr[1] = cc[maxIdx];

  rr[2] = viennacore::CrossProduct(rr[0], rr[1]);
  viennacore::Normalize(rr[0]);
  viennacore::Normalize(rr[1]);
  viennacore::Normalize(rr[2]);

  return rr;
}

__device__ void initializeRayDirection(viennaps::gpu::PerRayData *prd,
                                       const float power)
{
  // source direction
  auto r1 = getNextRand(&prd->RNGstate);
  auto r2 = getNextRand(&prd->RNGstate);
  const float ee = 2.f / (power + 1.f);
  const float tt = powf(r2, ee);
  prd->dir[0] = cosf(2 * M_PIf * r1) * sqrtf(1 - tt);
  prd->dir[1] = sinf(2 * M_PIf * r1) * sqrtf(1 - tt);
  prd->dir[2] = -1.f * sqrtf(tt);
  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayDirection(viennaps::gpu::PerRayData *prd, const float power,
                       const std::array<viennacore::Vec3Df, 3> &basis)
{
  // source direction
  do
  {
    auto r1 = getNextRand(&prd->RNGstate);
    auto r2 = getNextRand(&prd->RNGstate);
    const float ee = 2.f / (power + 1.f);
    const float tt = powf(r2, ee);
    viennacore::Vec3Df rndDirection;
    rndDirection[0] = sqrtf(tt);
    rndDirection[1] = cosf(2 * M_PIf * r1) * sqrtf(1 - tt);
    rndDirection[2] = sinf(2 * M_PIf * r1) * sqrtf(1 - tt);

    prd->dir[0] = basis[0][0] * rndDirection[0] + basis[1][0] * rndDirection[1] +
                  basis[2][0] * rndDirection[2];
    prd->dir[1] = basis[0][1] * rndDirection[0] + basis[1][1] * rndDirection[1] +
                  basis[2][1] * rndDirection[2];
    prd->dir[2] = basis[0][2] * rndDirection[0] + basis[1][2] * rndDirection[1] +
                  basis[2][2] * rndDirection[2];
  } while (prd->dir[2] >= 0.f);

  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayPosition(viennaps::gpu::PerRayData *prd,
                      viennaps::gpu::LaunchParams<float> *launchParams)
{
  auto rx = getNextRand(&prd->RNGstate);
  auto ry = getNextRand(&prd->RNGstate);
  prd->pos[0] = launchParams->source.minPoint[0] +
                rx * (launchParams->source.maxPoint[0] -
                      launchParams->source.minPoint[0]);
  prd->pos[1] = launchParams->source.minPoint[1] +
                ry * (launchParams->source.maxPoint[1] -
                      launchParams->source.minPoint[1]);
  prd->pos[2] = launchParams->source.planeHeight;
}
#endif