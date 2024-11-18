#include <cuda.h>
#include <cuda_runtime.h>

#include <vcVectorUtil.hpp>

extern "C" __global__ void
normalize_surface_f(float *data, const viennacore::Vec3Df *vertex,
                    const viennacore::Vec3D<unsigned> *index,
                    const unsigned int numTriangles,
                    const float sourceArea, const size_t numRays,
                    const int numData)
{
  using namespace viennacore;
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride)
  {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx[0]];
    const auto &B = vertex[elIdx[1]];
    const auto &C = vertex[elIdx[2]];
    const auto area = Norm(CrossProduct(B - A, C - A)) / 2.f;
    if (area > 1e-6f)
      data[tidx] *= sourceArea / (area * (float)numRays);
    else
      data[tidx] = 0.f;
  }
}

extern "C" __global__ void
normalize_surface_d(double *data, const viennacore::Vec3Df *vertex,
                    const viennacore::Vec3D<unsigned> *index,
                    const unsigned int numTriangles,
                    const double sourceArea, const size_t numRays,
                    const int numData)
{
  using namespace viennacore;
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride)
  {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx[0]];
    const auto &B = vertex[elIdx[1]];
    const auto &C = vertex[elIdx[2]];
    const double area = Norm(CrossProduct(B - A, C - A)) / 2.;
    if (area > 1e-8)
      data[tidx] *= sourceArea / (area * (double)numRays);
    else
      data[tidx] = 0.;
  }
}