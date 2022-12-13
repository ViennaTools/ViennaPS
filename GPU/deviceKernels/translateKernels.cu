#include <cuda.h>

#include <utGDT.hpp>

#define THREADS_PER_BLOCK 512
#define NUM_BLOCKS 512

// extern "C" __global__ void refine_mesh_mark_points(const std::array<float, 3>
// *vertex,
//                                                    const std::array<unsigned,
//                                                    3> *index, int *replace,
//                                                    float minNodeDistance,
//                                                    unsigned int numElements)
// {
//   for (unsigned int bidx = blockIdx.x; bidx < numElements; bidx += gridDim.x)
//   {
//     for (unsigned int tidx = threadIdx.x; tidx < numElements;
//          tidx += blockDim.x)
//     {
//       if (tidx == bidx)
//         continue;
//       float nodeDist = abs(vertex[bidx][0] - vertex[bidx][0]) +
//                        abs(vertex[bidx][1] - vertex[bidx][1]) +
//                        abs(vertex[bidx][2] - vertex[bidx][2]);
//       if (nodeDist < minNodeDistance)
//       {
//         atomicAdd(&replace[bidx], 1);
//       }
//     }
//   }
// }

extern "C" __global__ void translate_to_point_cloud_mesh_f(
    const gdt::vec3f *vertex, const gdt::vec3i *index, const float *values,
    const std::array<float, 3> *point, float *pointValues, const float radius,
    const size_t numPoints, const unsigned int numTriangles,
    const unsigned int numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numPoints; bidx += gridDim.x)
  {
    // __syncthreads();
    cache[threadIdx.x] = 0;
    for (unsigned int tidx = threadIdx.x; tidx < numTriangles;
         tidx += blockDim.x)
    {
      const gdt::vec3i triangle = index[tidx];
      const gdt::vec3f center =
          (vertex[triangle.r] + vertex[triangle.s] + vertex[triangle.t]) / 3.f;
      if (gdt::length(center - point[bidx]) < radius)
      {
        for (unsigned int i = 0; i < numData; i++)
        {
          atomicAdd(&pointValues[bidx + i * numPoints],
                    values[tidx + i * numTriangles]);
        }
        cache[threadIdx.x]++;
      }
    }
    __syncthreads();

    int i = THREADS_PER_BLOCK / 2;
    while (i > 0)
    {
      if (threadIdx.x < i)
        cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2;
    }

    if (threadIdx.x == 0 && cache[0] > 1)
    {
      for (unsigned int i = 0; i < numData; i++)
      {
        pointValues[bidx + i * numPoints] /= cache[0];
      }
    }
  }
}

extern "C" __global__ void translate_to_point_cloud_mesh_d(
    const gdt::vec3f *vertex, const gdt::vec3i *index, const double *values,
    const std::array<double, 3> *point, double *pointValues,
    const double radius, const size_t numPoints,
    const unsigned int numTriangles, const unsigned int numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numPoints; bidx += gridDim.x)
  {
    // __syncthreads();
    cache[threadIdx.x] = 0;
    for (unsigned int tidx = threadIdx.x; tidx < numTriangles;
         tidx += blockDim.x)
    {
      const gdt::vec3i triangle = index[tidx];
      const gdt::vec3f center =
          (vertex[triangle.r] + vertex[triangle.s] + vertex[triangle.t]) / 3.f;
      if (gdt::length(center - point[bidx]) < radius)
      {
        for (unsigned int i = 0; i < numData; i++)
        {
          atomicAdd(&pointValues[bidx + i * numPoints],
                    values[tidx + i * numTriangles]);
        }
        cache[threadIdx.x]++;
      }
    }
    __syncthreads();

    int i = THREADS_PER_BLOCK / 2;
    while (i > 0)
    {
      if (threadIdx.x < i)
        cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2;
    }

    if (threadIdx.x == 0 && cache[0] > 1)
    {
      for (unsigned int i = 0; i < numData; i++)
      {
        pointValues[bidx + i * numPoints] /= cache[0];
      }
    }
  }
}

extern "C" __global__ void translate_to_point_cloud_mesh_closest_d(
    const gdt::vec3f *vertex, const gdt::vec3i *index, const double *values,
    const std::array<double, 3> *point, double *pointValues,
    const double radius, const size_t numPoints,
    const unsigned int numTriangles, const unsigned int numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numPoints; bidx += gridDim.x)
  {
    gdt::vec3f center =
        (vertex[index[0].r] + vertex[index[0].s] + vertex[index[0].t]) / 3.f;

    cache[threadIdx.x] = 0;
    float distance = gdt::length(center - point[bidx]);
    for (unsigned int tidx = threadIdx.x; tidx < numTriangles;
         tidx += blockDim.x)
    {
      center = (vertex[index[tidx].r] + vertex[index[tidx].s] +
                vertex[index[tidx].t]) /
               3.f;

      float newDistance = gdt::length(center - point[bidx]);
      if (newDistance < distance)
      {
        distance = newDistance;
        cache[threadIdx.x] = tidx;
      }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
      {
        const gdt::vec3i triangle_1 = index[cache[threadIdx.x + s]];
        const gdt::vec3i triangle_2 = index[cache[threadIdx.x]];

        const gdt::vec3f center_1 = (vertex[triangle_1.r] + vertex[triangle_1.s] +
                                     vertex[triangle_1.t]) /
                                    3.f;
        const gdt::vec3f center_2 = (vertex[triangle_2.r] + vertex[triangle_2.s] +
                                     vertex[triangle_2.t]) /
                                    3.f;

        if (gdt::length(center_1 - point[bidx]) <
            gdt::length(center_2 - point[bidx]))
        {
          cache[threadIdx.x] = cache[threadIdx.x + s];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
      for (unsigned i = 0; i < numData; i++)
      {
        pointValues[bidx + i * numPoints] = values[cache[0] + i * numTriangles];
      }
    }
  }
}

extern "C" __global__ void translate_to_point_cloud_mesh_closest_f(
    const gdt::vec3f *vertex, const gdt::vec3i *index, const float *values,
    const std::array<float, 3> *point, float *pointValues, const float radius,
    const size_t numPoints, const unsigned int numTriangles,
    const unsigned int numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numPoints; bidx += gridDim.x)
  {
    gdt::vec3f center =
        (vertex[index[0].r] + vertex[index[0].s] + vertex[index[0].t]) / 3.f;

    cache[threadIdx.x] = 0;
    float distance = gdt::length(center - point[bidx]);
    for (unsigned int tidx = threadIdx.x; tidx < numTriangles;
         tidx += blockDim.x)
    {
      center = (vertex[index[tidx].r] + vertex[index[tidx].s] +
                vertex[index[tidx].t]) /
               3.f;

      float newDistance = gdt::length(center - point[bidx]);
      if (newDistance < distance)
      {
        distance = newDistance;
        cache[threadIdx.x] = tidx;
      }
    }
    __syncthreads();

    int i = THREADS_PER_BLOCK / 2;
    while (i > 0)
    {
      if (threadIdx.x < i)
      {
        const gdt::vec3i triangle_1 = index[cache[threadIdx.x + i]];
        const gdt::vec3i triangle_2 = index[cache[threadIdx.x]];

        const gdt::vec3f center_1 = (vertex[triangle_1.r] + vertex[triangle_1.s] +
                                     vertex[triangle_1.t]) /
                                    3.f;
        const gdt::vec3f center_2 = (vertex[triangle_2.r] + vertex[triangle_2.s] +
                                     vertex[triangle_2.t]) /
                                    3.f;

        if (gdt::length(center_1 - point[bidx]) <
            gdt::length(center_2 - point[bidx]))
        {
          cache[threadIdx.x] = cache[threadIdx.x + i];
        }
      }
      __syncthreads();
      i /= 2;
    }

    if (threadIdx.x == 0)
    {
      for (unsigned i = 0; i < numData; i++)
      {
        pointValues[bidx + i * numPoints] = values[cache[0] + i * numTriangles];
      }
    }
  }
}

extern "C" __global__ void translate_from_point_cloud_mesh_d(
    const gdt::vec3f *vertex, const gdt::vec3i *index, double *cellValues,
    const std::array<double, 3> *point, const double *pointValues,
    const size_t numPoints, const unsigned int numTriangles,
    const unsigned numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numTriangles; bidx += gridDim.x)
  {
    const gdt::vec3i triangle = index[bidx];
    const gdt::vec3f center =
        (vertex[triangle.r] + vertex[triangle.s] + vertex[triangle.t]) / 3.f;

    cache[threadIdx.x] = 0;
    double distance = gdt::length(center - point[0]);
    for (unsigned int tidx = threadIdx.x; tidx < numPoints;
         tidx += blockDim.x)
    {
      double newDistance = gdt::length(center - point[tidx]);
      if (newDistance < distance)
      {
        distance = newDistance;
        cache[threadIdx.x] = tidx;
      }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
      {
        if (gdt::length(center - point[cache[threadIdx.x + s]]) <
            gdt::length(center - point[cache[threadIdx.x]]))
        {
          cache[threadIdx.x] = cache[threadIdx.x + s];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
      for (unsigned i = 0; i < numData; i++)
      {
        cellValues[bidx + i * numTriangles] =
            pointValues[cache[0] + i * numPoints];
      }
    }
  }
}

extern "C" __global__ void translate_from_point_cloud_mesh_f(
    const gdt::vec3f *vertex, const gdt::vec3i *index, float *cellValues,
    const std::array<float, 3> *point, const float *pointValues,
    const size_t numPoints, const unsigned int numTriangles,
    const unsigned numData)
{
  __shared__ int cache[THREADS_PER_BLOCK];

  for (unsigned int bidx = blockIdx.x; bidx < numTriangles; bidx += gridDim.x)
  {
    const gdt::vec3i triangle = index[bidx];
    const gdt::vec3f center =
        (vertex[triangle.r] + vertex[triangle.s] + vertex[triangle.t]) / 3.f;

    cache[threadIdx.x] = 0;
    float distance = gdt::length(center - point[0]);
    for (unsigned int tidx = threadIdx.x; tidx < numPoints;
         tidx += blockDim.x)
    {
      float newDistance = gdt::length(center - point[tidx]);
      if (newDistance < distance)
      {
        distance = newDistance;
        cache[threadIdx.x] = tidx;
      }
    }
    __syncthreads();

    int i = THREADS_PER_BLOCK / 2;
    while (i > 0)
    {
      if (threadIdx.x < i)
      {
        if (gdt::length(center - point[cache[threadIdx.x + i]]) <
            gdt::length(center - point[cache[threadIdx.x]]))
        {
          cache[threadIdx.x] = cache[threadIdx.x + i];
        }
      }
      __syncthreads();
      i /= 2;
    }

    if (threadIdx.x == 0)
    {
      for (unsigned i = 0; i < numData; i++)
      {
        cellValues[bidx + i * numTriangles] =
            pointValues[cache[0] + i * numPoints];
      }
    }
  }
}
