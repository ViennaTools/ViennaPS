#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

#include <utGDT.hpp>

template <typename T> struct Particle {
  std::array<T, 3> position;
  std::array<T, 3> direction;
  T energy;
  T distance;
  int cellId;
  int scattered;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &o, const std::array<T, 2> &v) {
  o << "(" << v[0] << "," << v[1] << ")";
  return o;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &o, const std::array<T, 3> &v) {
  o << "(" << v[0] << "," << v[1] << "," << v[2] << ")";
  return o;
}

namespace aops {
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
std::array<T, 3> crossProd(const std::array<T, 3> &pVecA,
                           const std::array<T, 3> &pVecB) {
  std::array<T, 3> rr;
  rr[0] = pVecA[1] * pVecB[2] - pVecA[2] * pVecB[1];
  rr[1] = pVecA[2] * pVecB[0] - pVecA[0] * pVecB[2];
  rr[2] = pVecA[0] * pVecB[1] - pVecA[1] * pVecB[0];
  return rr;
}
} // namespace aops

// -------------------------------------------------------
// binary operators
// -------------------------------------------------------
#define _define_operator(op)                                                   \
  /* vec op vec */                                                             \
  template <typename T>                                                        \
  inline __both__ std::array<T, 2> operator op(const std::array<T, 2> &a,      \
                                               const std::array<T, 2> &b) {    \
    return std::array<T, 2>(a[0] op b[0], a[1] op b[1]);                       \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ std::array<T, 3> operator op(const std::array<T, 3> &a,      \
                                               const std::array<T, 3> &b) {    \
    return std::array<T, 3>(a[0] op b[0], a[1] op b[1], a[2] op b[2]);         \
  }                                                                            \
                                                                               \
  template <typename T, typename OT>                                           \
  inline __both__ std::array<T, 3> operator op(const std::array<T, 3> &a,      \
                                               const std::array<OT, 3> &b) {   \
    return std::array<T, 3>(a[0] op b[0], a[1] op b[1], a[2] op b[2]);         \
  }                                                                            \
                                                                               \
  /* vec op scalar */                                                          \
  template <typename T>                                                        \
  inline __both__ std::array<T, 2> operator op(const std::array<T, 2> &a,      \
                                               const T &b) {                   \
    return std::array<T, 2>(a[0] op b, a[1] op b);                             \
  }                                                                            \
                                                                               \
  /* scalar op vec */                                                          \
  template <typename T>                                                        \
  inline __both__ std::array<T, 2> operator op(const T &a,                     \
                                               const std::array<T, 2> &b) {    \
    return std::array<T, 2>(a op b[0], a op b[1]);                             \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ std::array<T, 3> operator op(const T &a,                     \
                                               const std::array<T, 3> &b) {    \
    return std::array<T, 3>(a op b[0], a op b[1], a op b[2]);                  \
  }

_define_operator(*);
_define_operator(/);
_define_operator(+);
_define_operator(-);

#undef _define_operator
