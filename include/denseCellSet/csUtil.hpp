#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

template <typename T> using csPair = std::array<T, 2>;

template <typename T> using csTriple = std::array<T, 3>;

template <typename T> struct csVolumeParticle {
  csTriple<T> position;
  csTriple<T> direction;
  T energy;
  T distance;
  int cellId;
  int scattered;
};

namespace csUtil {
template <typename T> void printTriple(const csTriple<T> &p) {
  std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]\n";
}

template <typename T, size_t D>
inline T dot(const std::array<T, D> &v1, const std::array<T, D> &v2) {
  T res = 0;
  for (int i = 0; i < D; i++)
    res += v1[i] * v2[i];
  return res;
}

template <typename T, size_t D>
inline void mult(std::array<T, D> &v1, const T fac) {
  for (int i = 0; i < D; i++)
    v1[i] *= fac;
}

template <typename T, size_t D>
inline std::array<T, D> multNew(const std::array<T, D> &v1, const T fac) {
  std::array<T, D> res;
  for (int i = 0; i < D; i++)
    res[i] = v1[i] * fac;
  return res;
}

template <typename T, size_t D>
inline void add(std::array<T, D> &vec, const std::array<T, D> &vec2) {
  for (size_t i = 0; i < D; i++)
    vec[i] += vec2[i];
}

template <typename T, size_t D>
inline void sub(std::array<T, D> &v1, const std::array<T, D> &v2) {
  for (int i = 0; i < D; i++)
    v1[i] -= v2[i];
}

template <typename T, size_t D>
inline void multAdd(std::array<T, D> &result, const std::array<T, D> &mult,
                    const std::array<T, D> &add, const T fac) {
  for (int i = 0; i < D; i++)
    result[i] = add[i] + mult[i] * fac;
}

template <typename T, size_t D>
inline T distance(const std::array<T, D> &p1, const std::array<T, D> &p2) {
  T res = 0;
  for (int i = 0; i < D; i++)
    res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  return std::sqrt(res);
}

template <typename T, size_t D> inline T norm(const std::array<T, D> &vec) {
  T res = 0;
  for (int i = 0; i < D; i++)
    res += vec[i] * vec[i];
  return std::sqrt(res);
}

template <typename T, size_t D> void normalize(std::array<T, D> &vec) {
  T vecNorm = 1. / norm(vec);
  if (vecNorm == 1.)
    return;
  std::for_each(vec.begin(), vec.end(),
                [&vecNorm](T &entry) { entry *= vecNorm; });
}

template <typename T, size_t D>
void scaleToLength(std::array<T, D> &vec, const T length) {
  const auto vecLength = norm(vec);
  for (size_t i = 0; i < D; i++)
    vec[i] *= length / vecLength;
}

template <typename T>
csTriple<T> crossProd(const csTriple<T> &pVecA, const csTriple<T> &pVecB) {
  csTriple<T> rr;
  rr[0] = pVecA[1] * pVecB[2] - pVecA[2] * pVecB[1];
  rr[1] = pVecA[2] * pVecB[0] - pVecA[0] * pVecB[2];
  rr[2] = pVecA[0] * pVecB[1] - pVecA[1] * pVecB[0];
  return rr;
}

#ifdef ARCH_X86
[[nodiscard]] static inline float DotProductSse(__m128 const &x,
                                                __m128 const &y) {
  // __m128 mulRes;
  // mulRes = _mm_mul_ps(x, y);
  // return SumSse(mulRes);
  return _mm_cvtss_f32(_mm_dp_ps(x, y, 0x77));
}

[[nodiscard]] static inline float SumSse(__m128 const &v) {
  __m128 shufReg, sumsReg;
  // Calculates the sum of SSE Register -
  // https://stackoverflow.com/a/35270026/195787
  shufReg = _mm_movehdup_ps(v); // Broadcast elements 3,1 to 2,0
  sumsReg = _mm_add_ps(v, shufReg);
  shufReg = _mm_movehl_ps(shufReg, sumsReg); // High Half -> Low Half
  sumsReg = _mm_add_ss(sumsReg, shufReg);
  return _mm_cvtss_f32(sumsReg); // Result in the lower part of the SSE Register
}

[[nodiscard]] inline static __m128 CrossProductSse(__m128 const &vec0,
                                                   __m128 const &vec1) {
  // from geometrian.com/programming/tutorials/cross-product/index.php
  __m128 tmp0 = _mm_shuffle_ps(vec0, vec0, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 tmp1 = _mm_shuffle_ps(vec1, vec1, _MM_SHUFFLE(3, 1, 0, 2));
  __m128 tmp2 = _mm_mul_ps(tmp0, vec1);
  __m128 tmp3 = _mm_mul_ps(tmp0, tmp1);
  __m128 tmp4 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(3, 0, 2, 1));
  return _mm_sub_ps(tmp3, tmp4);
}

// Norm of 3D Vector using SSE
// fastcpp.blogspot.com/2012/02/calculating-length-of-3d-vector-using.html
[[nodiscard]] static inline float NormSse(__m128 const &v) {
  return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(v, v, 0x71)));
}

[[nodiscard]] static inline __m128 NormalizeSse(__m128 const &v) {
  __m128 inverse_norm = _mm_rsqrt_ps(_mm_dp_ps(v, v, 0x77));
  return _mm_mul_ps(v, inverse_norm);
}

[[nodiscard]] static inline __m128 NormalizeAccurateSse(__m128 const &v) {
  __m128 norm = _mm_sqrt_ps(_mm_dp_ps(v, v, 0x7F));
  return _mm_div_ps(v, norm);
}

template <typename T> [[nodiscard]] rayTriple<T> ConvertSse(__m128 const &vec) {
  alignas(16) float result[4];
  _mm_store_ps(&result[0], vec);
  return csTriple<T>{result[0], result[1], result[2]};
}
#endif
} // namespace csUtil
