#ifndef UT_GDT_HPP
#define UT_GDT_HPP

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <vector>
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#else
#include <cmath>
#endif
#include <algorithm>
#ifdef __GNUC__
#include <stdint.h>
#endif
#include <stdexcept>

#if defined(__CUDACC__)
#define __gdt_device __device__
#define __gdt_host __host__
#else
#define __gdt_device /* ignore */
#define __gdt_host   /* ignore */
#endif

#define __both__ __gdt_host __gdt_device

#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

/*! namespace gdt GPU Developer Toolbox */
namespace gdt {

#ifdef __CUDACC__
using ::max;
using ::min;
// inline __both__ float abs(float f)      { return fabsf(f); }
// inline __both__ double abs(double f)    { return fabs(f); }
using std::abs;
// inline __both__ float sin(float f) { return ::sinf(f); }
// inline __both__ double sin(double f) { return ::sin(f); }
// inline __both__ float cos(float f) { return ::cosf(f); }
// inline __both__ double cos(double f) { return ::cos(f); }
#else
using std::abs;
using std::max;
using std::min;
#endif

// inline __both__ float abs(float f)      { return fabsf(f); }
// inline __both__ double abs(double f)    { return fabs(f); }
inline __both__ float rcp(float f) { return 1.f / f; }
inline __both__ double rcp(double d) { return 1. / d; }

inline __both__ int32_t divRoundUp(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}
inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}
inline __both__ int64_t divRoundUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}
inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) {
  return (a + b - 1) / b;
}

#ifdef __CUDACC__
using ::sin; // this is the double version
// inline __both__ float sin(float f) { return ::sinf(f); }
using ::cos; // this is the double version
// inline __both__ float cos(float f) { return ::cosf(f); }
#else
using ::cos; // this is the double version
using ::sin; // this is the double version
#endif

namespace overloaded {
/* move all those in a special namespace so they will never get
   included - and thus, conflict with, the default namesapce */
inline __both__ float sqrt(const float f) { return ::sqrtf(f); }
inline __both__ double sqrt(const double d) { return ::sqrt(d); }
} // namespace overloaded

#ifdef __WIN32__
#define osp_snprintf sprintf_s
#else
#define osp_snprintf snprintf
#endif

/*! added pretty-print function for large numbers, printing 10000000 as "10M"
 * instead */
inline std::string prettyDouble(const double val) {
  const double absVal = abs(val);
  char result[1000];

  if (absVal >= 1e+18f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e18f, 'E');
  else if (absVal >= 1e+15f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e15f, 'P');
  else if (absVal >= 1e+12f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e12f, 'T');
  else if (absVal >= 1e+09f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e09f, 'G');
  else if (absVal >= 1e+06f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e06f, 'M');
  else if (absVal >= 1e+03f)
    osp_snprintf(result, 1000, "%.1f%c", val / 1e03f, 'k');
  else if (absVal <= 1e-12f)
    osp_snprintf(result, 1000, "%.1f%c", val * 1e15f, 'f');
  else if (absVal <= 1e-09f)
    osp_snprintf(result, 1000, "%.1f%c", val * 1e12f, 'p');
  else if (absVal <= 1e-06f)
    osp_snprintf(result, 1000, "%.1f%c", val * 1e09f, 'n');
  else if (absVal <= 1e-03f)
    osp_snprintf(result, 1000, "%.1f%c", val * 1e06f, 'u');
  else if (absVal <= 1e-00f)
    osp_snprintf(result, 1000, "%.1f%c", val * 1e03f, 'm');
  else
    osp_snprintf(result, 1000, "%f", (float)val);

  return result;
}

inline std::string prettyNumber(const size_t s) {
  char buf[1000];
  if (s >= (1024LL * 1024LL * 1024LL * 1024LL)) {
    osp_snprintf(buf, 1000, "%.2fT", s / (1024.f * 1024.f * 1024.f * 1024.f));
  } else if (s >= (1024LL * 1024LL * 1024LL)) {
    osp_snprintf(buf, 1000, "%.2fG", s / (1024.f * 1024.f * 1024.f));
  } else if (s >= (1024LL * 1024LL)) {
    osp_snprintf(buf, 1000, "%.2fM", s / (1024.f * 1024.f));
  } else if (s >= (1024LL)) {
    osp_snprintf(buf, 1000, "%.2fK", s / (1024.f));
  } else {
    osp_snprintf(buf, 1000, "%zi", s);
  }
  return buf;
}

// inline double getCurrentTime() {
// #ifdef _WIN32
//   SYSTEMTIME tp;
//   GetSystemTime(&tp);
//   return double(tp.wSecond) + double(tp.wMilliseconds) / 1E3;
// #else
//   struct timeval tp;
//   gettimeofday(&tp, nullptr);
//   return double(tp.tv_sec) + double(tp.tv_usec) / 1E6;
// #endif
// }

template <typename TimeUnit> static uint64_t timeStampNow() {
  return std::chrono::duration_cast<TimeUnit>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline bool hasSuffix(const std::string &s, const std::string &suffix) {
  return s.substr(s.size() - suffix.size()) == suffix;
}

template <typename T> struct long_type_of { typedef T type; };
template <> struct long_type_of<int32_t> { typedef int64_t type; };
template <> struct long_type_of<uint32_t> { typedef uint64_t type; };

template <typename T, int N> struct vec_t { T t[N]; };

template <typename ScalarTypeA, typename ScalarTypeB> struct BinaryOpResultType;

// Binary Result type: scalar type with itself always returns same type
template <typename ScalarType>
struct BinaryOpResultType<ScalarType, ScalarType> {
  typedef ScalarType type;
};

template <> struct BinaryOpResultType<int, float> { typedef float type; };
template <> struct BinaryOpResultType<float, int> { typedef float type; };
template <> struct BinaryOpResultType<unsigned int, float> {
  typedef float type;
};
template <> struct BinaryOpResultType<float, unsigned int> {
  typedef float type;
};

template <> struct BinaryOpResultType<int, double> { typedef double type; };
template <> struct BinaryOpResultType<double, int> { typedef double type; };
template <> struct BinaryOpResultType<unsigned int, double> {
  typedef double type;
};
template <> struct BinaryOpResultType<double, unsigned int> {
  typedef double type;
};

// ------------------------------------------------------------------
// vec2
// ------------------------------------------------------------------
template <typename T> struct vec_t<T, 2> {
  enum { dims = 2 };
  typedef T scalar_t;

  inline __both__ vec_t() {}
  inline __both__ vec_t(const T &t) : x(t), y(t) {}
  inline __both__ vec_t(const T &x, const T &y) : x(x), y(y) {}
  inline __both__ vec_t(const std::array<T, 2> &o) : x(o[0]), y(o[1]) {}
#ifdef __CUDACC__
  inline __both__ vec_t(const float2 v) : x(v.x), y(v.y) {}
  inline __both__ vec_t(const int2 v) : x(v.x), y(v.y) {}
  inline __both__ vec_t(const uint2 v) : x(v.x), y(v.y) {}

  inline __both__ operator float2() const { return make_float2(x, y); }
  inline __both__ operator int2() const { return make_int2(x, y); }
  inline __both__ operator uint2() const { return make_uint2(x, y); }
#endif

  /*! assignment operator */
  inline __both__ vec_t<T, 2> &operator=(const vec_t<T, 2> &other) {
    this->x = other.x;
    this->y = other.y;
    return *this;
  }
  inline __both__ vec_t<T, 2> &operator=(const vec_t<T, 3> &other) {
    this->x = other.x;
    this->y = other.y;
    return *this;
  }
  inline __both__ vec_t<T, 2> &operator=(const std::array<T, 2> &other) {
    this->x = other[0];
    this->y = other[1];
    return *this;
  }
  inline __both__ vec_t<T, 2> &operator=(const std::array<T, 3> &other) {
    this->x = other[0];
    this->y = other[1];
    return *this;
  }

  template <typename OT>
  inline __both__ vec_t<T, 2> &operator=(const std::array<OT, 2> &other) {
    this->x = (T)other[0];
    this->y = (T)other[1];
    return *this;
  }
  template <typename OT>
  inline __both__ vec_t<T, 2> &operator=(const std::array<OT, 3> &other) {
    this->x = (T)other[0];
    this->y = (T)other[1];
    return *this;
  }

  /*! construct 2-vector from 2-vector of another type */
  template <typename OT>
  inline __both__ explicit vec_t(const vec_t<OT, 2> &o)
      : x((T)o.x), y((T)o.y) {}

  template <typename OT>
  inline __both__ explicit vec_t(const std::array<OT, 2> &o)
      : x((T)o[0]), y((T)o[1]) {}

  inline __both__ T &operator[](size_t dim) { return (&x)[dim]; }
  inline __both__ const T &operator[](size_t dim) const { return (&x)[dim]; }

  union {
    struct {
      T x, y;
    };
    struct {
      T s, t;
    };
    struct {
      T u, v;
    };
  };
};

// ------------------------------------------------------------------
// vec3
// ------------------------------------------------------------------
template <typename T> struct vec_t<T, 3> {
  enum { dims = 3 };
  typedef T scalar_t;

  inline __both__ vec_t() {}
  inline __both__ vec_t(const T &t) : x(t), y(t), z(t) {}
  inline __both__ vec_t(const T &_x, const T &_y, const T &_z)
      : x(_x), y(_y), z(_z) {}
  inline __both__ vec_t(const std::array<T, 3> &o)
      : x(o[0]), y(o[1]), z(o[2]) {}
#ifdef __CUDACC__
  inline __both__ vec_t(const int3 &v) : x(v.x), y(v.y), z(v.z) {}
  inline __both__ vec_t(const uint3 &v) : x(v.x), y(v.y), z(v.z) {}
  inline __both__ vec_t(const float3 &v) : x(v.x), y(v.y), z(v.z) {}
  inline __both__ operator float3() const { return make_float3(x, y, z); }
  inline __both__ operator int3() const { return make_int3(x, y, z); }
  inline __both__ operator uint3() const { return make_uint3(x, y, z); }
#endif
  inline __both__ explicit vec_t(const vec_t<T, 4> &v);
  /*! construct 3-vector from 3-vector of another type */
  template <typename OT>
  inline __both__ explicit vec_t(const vec_t<OT, 3> &o)
      : x((T)o.x), y((T)o.y), z((T)o.z) {}
  template <typename OT>
  inline __both__ explicit vec_t(const std::array<OT, 3> &o)
      : x((T)o[0]), y((T)o[1]), z((T)o[2]) {}
  template <typename OT>
  inline __both__ vec_t<T, 3> &operator=(const std::array<OT, 3> &other) {
    this->x = (T)other[0];
    this->y = (T)other[1];
    this->z = (T)other[2];
    return *this;
  }

  /*! swizzle ... */
  inline __both__ vec_t<T, 3> yzx() const { return vec_t<T, 3>(y, z, x); }

  /*! assignment operator */
  inline __both__ vec_t<T, 3> &operator=(const vec_t<T, 3> &other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    return *this;
  }
  inline __both__ vec_t<T, 3> &operator=(const vec_t<T, 2> &other) {
    this->x = other.x;
    this->y = other.y;
    this->z = T(0);
    return *this;
  }
  inline __both__ vec_t<T, 3> &operator=(const std::array<T, 3> &other) {
    this->x = other[0];
    this->y = other[1];
    this->z = other[2];
    return *this;
  }
  inline __both__ vec_t<T, 3> &operator=(const std::array<T, 2> &other) {
    this->x = other[0];
    this->y = other[1];
    this->z = T(0);
    return *this;
  }

  inline __both__ T &operator[](size_t dim) { return (&x)[dim]; }
  inline __both__ const T &operator[](size_t dim) const { return (&x)[dim]; }

  template <typename OT, typename Lambda>
  static inline __both__ vec_t<T, 3> make_from(const vec_t<OT, 3> &v,
                                               const Lambda &lambda) {
    return vec_t<T, 3>(lambda(v.x), lambda(v.y), lambda(v.z));
  }

  union {
    struct {
      T x, y, z;
    };
    struct {
      T r, s, t;
    };
    struct {
      T u, v, w;
    };
  };
};

// =======================================================
// default functions
// =======================================================

template <typename T>
inline __both__ typename long_type_of<T>::type area(const vec_t<T, 2> &v) {
  return (typename long_type_of<T>::type)(v.x) *
         (typename long_type_of<T>::type)(v.y);
}

template <typename T>
inline __both__ typename long_type_of<T>::type volume(const vec_t<T, 3> &v) {
  return (typename long_type_of<T>::type)(v.x) *
         (typename long_type_of<T>::type)(v.y) *
         (typename long_type_of<T>::type)(v.z);
}

template <typename T>
inline __both__ typename long_type_of<T>::type area(const vec_t<T, 3> &v) {
  return T(2) * ((typename long_type_of<T>::type)(v.x) * v.y +
                 (typename long_type_of<T>::type)(v.y) * v.z +
                 (typename long_type_of<T>::type)(v.z) * v.x);
}

template <typename T> inline __both__ vec_t<T, 3> neg(const vec_t<T, 3> &a) {
  return vec_t<T, 3>(-a.x, -a.y, -a.z);
}

/*! vector cross product */
template <typename T>
inline __both__ vec_t<T, 3> cross(const vec_t<T, 3> &a, const vec_t<T, 3> &b) {
  return vec_t<T, 3>(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x,
                     a.x * b.y - b.x * a.y);
}

/*! vector dot product */
template <typename T, typename OT>
inline __both__ T dot(const vec_t<T, 3> &a, const vec_t<OT, 3> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*! vector normalization */
template <typename T>
inline __both__ vec_t<T, 3> normalize(const vec_t<T, 3> &v) {
  return v * 1.f / gdt::overloaded::sqrt(dot(v, v));
}

/*! vector length */
template <typename T> inline __both__ T length(const vec_t<T, 3> &v) {
  return gdt::overloaded::sqrt(dot(v, v));
}

template <typename T>
inline __gdt_host std::ostream &operator<<(std::ostream &o,
                                           const vec_t<T, 2> &v) {
  o << "(" << v.x << "," << v.y << ")";
  return o;
}

template <typename T>
inline __gdt_host std::ostream &operator<<(std::ostream &o,
                                           const vec_t<T, 3> &v) {
  o << "(" << v.x << "," << v.y << "," << v.z << ")";
  return o;
}

// -------------------------------------------------------
// binary operators
// -------------------------------------------------------
#define _define_operator(op)                                                   \
  /* vec op vec */                                                             \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 2> operator op(const vec_t<T, 2> &a,                \
                                          const vec_t<T, 2> &b) {              \
    return vec_t<T, 2>(a.x op b.x, a.y op b.y);                                \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 3> operator op(const vec_t<T, 3> &a,                \
                                          const vec_t<T, 3> &b) {              \
    return vec_t<T, 3>(a.x op b.x, a.y op b.y, a.z op b.z);                    \
  }                                                                            \
                                                                               \
  template <typename T, typename OT>                                           \
  inline __both__ vec_t<T, 3> operator op(const vec_t<T, 3> &a,                \
                                          const vec_t<OT, 3> &b) {             \
    return vec_t<T, 3>(a.x op b.x, a.y op b.y, a.z op b.z);                    \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 3> operator op(const vec_t<T, 3> &a,                \
                                          const std::array<T, 3> &b) {         \
    return vec_t<T, 3>(a.x op b[0], a.y op b[1], a.z op b[2]);                 \
  }                                                                            \
                                                                               \
  template <typename T, typename OT>                                           \
  inline __both__ vec_t<T, 3> operator op(const vec_t<T, 3> &a,                \
                                          const std::array<OT, 3> &b) {        \
    return vec_t<T, 3>(a.x op b[0], a.y op b[1], a.z op b[2]);                 \
  }                                                                            \
                                                                               \
  /* vec op scalar */                                                          \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 2> operator op(const vec_t<T, 2> &a, const T &b) {  \
    return vec_t<T, 2>(a.x op b, a.y op b);                                    \
  }                                                                            \
                                                                               \
  template <typename T1, typename T2>                                          \
  inline __both__ vec_t<typename BinaryOpResultType<T1, T2>::type, 3>          \
  operator op(const vec_t<T1, 3> &a, const T2 &b) {                            \
    return vec_t<typename BinaryOpResultType<T1, T2>::type, 3>(                \
        a.x op b, a.y op b, a.z op b);                                         \
  }                                                                            \
                                                                               \
  /* scalar op vec */                                                          \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 2> operator op(const T &a, const vec_t<T, 2> &b) {  \
    return vec_t<T, 2>(a op b.x, a op b.y);                                    \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline __both__ vec_t<T, 3> operator op(const T &a, const vec_t<T, 3> &b) {  \
    return vec_t<T, 3>(a op b.x, a op b.y, a op b.z);                          \
  }

_define_operator(*);
_define_operator(/);
_define_operator(+);
_define_operator(-);

#undef _define_operator

// =======================================================
// default instantiations
// =======================================================

#define _define_vec_types(T, t)                                                \
  using vec2##t = vec_t<T, 2>;                                                 \
  using vec3##t = vec_t<T, 3>;

_define_vec_types(int8_t, c);
_define_vec_types(int16_t, s);
_define_vec_types(int32_t, i);
_define_vec_types(int64_t, l);
_define_vec_types(uint8_t, uc);
_define_vec_types(uint16_t, us);
_define_vec_types(uint32_t, ui);
_define_vec_types(uint64_t, ul);
_define_vec_types(float, f);
_define_vec_types(double, d);

#undef _define_vec_types

} // namespace gdt

#endif // UT_GDT_HPP