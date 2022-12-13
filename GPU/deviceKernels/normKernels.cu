#include <cuda.h>
#include <cuda_runtime.h>

#include <utGDT.hpp>

extern "C" __global__ void
normalize_surface_f(float *data, const gdt::vec3f *vertex,
                    const gdt::vec3i *index, const unsigned int numTriangles,
                    const float sourceArea, const size_t numRays,
                    const int numData)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride)
  {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx.x];
    const auto &B = vertex[elIdx.y];
    const auto &C = vertex[elIdx.z];
    const auto area = gdt::length(gdt::cross(B - A, C - A)) / 2.f;
    if (area > 1e-7f)
      data[tidx] *= sourceArea / (area * (float)numRays);
    else
      data[tidx] = 0.f;
  }
}

extern "C" __global__ void
normalize_surface_d(double *data, const gdt::vec3f *vertex,
                    const gdt::vec3i *index, const unsigned int numTriangles,
                    const double sourceArea, const size_t numRays,
                    const int numData)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride)
  {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx.x];
    const auto &B = vertex[elIdx.y];
    const auto &C = vertex[elIdx.z];
    const auto area = gdt::length(gdt::cross(B - A, C - A)) / 2.;
    if (area > 1e-8)
      data[tidx] *= sourceArea / (area * (double)numRays);
    else
      data[tidx] = 0.;
  }
}