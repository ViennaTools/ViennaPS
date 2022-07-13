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

template <typename T> using csQuadruple = std::array<T, 4>;

template <typename T> struct Particle {
  csTriple<T> position;
  csTriple<T> direction;
  T energy;
  T distance;
  int cellId;
};

template <typename T> void printTriple(const csTriple<T> &p) {
  std::cout << "[" << p[0] << ", " << p[1] << ", " << p[2] << "]\n";
}

template <typename T>
inline T dot(const csTriple<T> &v1, const csTriple<T> &v2) {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <typename T> inline void mult(csTriple<T> &v1, const T fac) {
  v1[0] *= fac;
  v1[1] *= fac;
  v1[2] *= fac;
}

template <typename T>
inline csTriple<T> multNew(const csTriple<T> &v1, const T fac) {
  return {v1[0] * fac, v1[1] * fac, v1[2] * fac};
}

template <typename T> inline void sub(csTriple<T> &v1, const csTriple<T> &v2) {
  v1[0] -= v2[0];
  v1[1] -= v2[1];
  v1[2] -= v2[2];
}

template <typename T>
void multAdd(csTriple<T> &result, const csTriple<T> &mult,
             const csTriple<T> &add, const T fac) {
  result[0] = add[0] + mult[0] * fac;
  result[1] = add[1] + mult[1] * fac;
  result[2] = add[2] + mult[2] * fac;
}

template <typename T>
inline T distance(const csTriple<T> &p1, const csTriple<T> &p2) {
  return std::sqrt(std::pow(p1[0] - p2[0], 2.) + std::pow(p1[1] - p2[1], 2.) +
                   std::pow(p1[2] - p2[2], 2.));
}

template <typename T> inline T norm(const csTriple<T> &vec) {
  return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

template <typename T, size_t D> void normalize(std::array<T, D> &vec) {
  T vecNorm = 1. / norm(vec);
  if (vecNorm == 1.)
    return;
  std::for_each(vec.begin(), vec.end(),
                [&vecNorm](T &entry) { entry *= vecNorm; });
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
