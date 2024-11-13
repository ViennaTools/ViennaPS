#pragma once

#include <curtLaunchParams.hpp>

#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#ifdef __GNUC__
#include <stdint.h>
#endif
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#if defined(_MSC_VER)
#define GDT_DLL_EXPORT __declspec(dllexport)
#define GDT_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#define GDT_DLL_EXPORT __attribute__((visibility("default")))
#define GDT_DLL_IMPORT __attribute__((visibility("default")))
#else
#define GDT_DLL_EXPORT
#define GDT_DLL_IMPORT
#endif

#ifndef PI_F
#define PI_F 3.14159265
#endif

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ unsigned int
getIdx(int particleIdx, int dataIdx,
       viennaps::gpu::LaunchParams<T> *launchParams) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < particleIdx; i++)
    offset += launchParams->dataPerParticle[i];
  offset = (offset + dataIdx) * launchParams->numElements;
  return offset + optixGetPrimitiveIndex();
}
#endif
