#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <vector>

#include <curtChecks.hpp>

/// simple wrapper for creating, and managing a device-side CUDA buffer
struct utCudaBuffer {
  inline CUdeviceptr d_pointer() const { return (CUdeviceptr)d_ptr; }

  // re-size buffer to given number of bytes
  void resize(size_t size) {
    if (d_ptr)
      free();
    alloc(size);
  }

  template <typename T> void alloc_and_init(size_t size, const T init) {
    if (d_ptr)
      free();
    sizeInBytes = size * sizeof(T);
    CUDA_CHECK(Malloc((void **)&d_ptr, sizeInBytes));
    CUDA_CHECK(Memset(d_ptr, init, sizeInBytes));
  }

  template <typename T> void set(size_t count, const T init) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(Memset(d_ptr, init, sizeInBytes));
  }

  // allocate to given number of bytes
  void alloc(size_t size) {
    if (d_ptr)
      free();
    this->sizeInBytes = size;
    CUDA_CHECK(Malloc((void **)&d_ptr, sizeInBytes));
  }

  // free allocated memory
  void free() {
    CUDA_CHECK(Free(d_ptr));
    d_ptr = nullptr;
    sizeInBytes = 0;
  }

  template <typename T> void alloc_and_upload(const std::vector<T> &vt) {
    alloc(vt.size() * sizeof(T));
    upload((const T *)vt.data(), vt.size());
  }

  template <typename T> void alloc_and_upload_single(const T &vt) {
    alloc(sizeof(T));
    upload(&vt, 1);
  }

  template <typename T> void upload(const T *t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(
        Memcpy(d_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <typename T> void download(T *t, size_t count) {
    assert(d_ptr != nullptr);
    assert(sizeInBytes == count * sizeof(T));
    CUDA_CHECK(
        Memcpy((void *)t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  size_t sizeInBytes{0};
  void *d_ptr{nullptr};
};
