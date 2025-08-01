#pragma once

#ifdef VIENNAPS_USE_PRECOMPILED

#define PS_PRECOMPILE_PRECISION_DIMENSION(className)                           \
  typedef className<double, 2> className##_double_2;                           \
  typedef className<double, 3> className##_double_3;                           \
  typedef className<float, 2> className##_float_2;                             \
  typedef className<float, 3> className##_float_3;                             \
  extern template class className<double, 2>;                                  \
  extern template class className<double, 3>;                                  \
  extern template class className<float, 2>;                                   \
  extern template class className<float, 3>;

#define PS_PRECOMPILE_PRECISION(className)                                     \
  typedef className<double> className##_double;                                \
  typedef className<float> className##_float;                                  \
  extern template class className<double>;                                     \
  extern template class className<float>;

#else

// do nothing if we use header only
#define PS_PRECOMPILE_PRECISION_DIMENSION(className)                           \
  typedef className<double, 2> className##_double_2;                           \
  typedef className<double, 3> className##_double_3;                           \
  typedef className<float, 2> className##_float_2;                             \
  typedef className<float, 3> className##_float_3;

#define PS_PRECOMPILE_PRECISION(className)                                     \
  typedef className<double> className##_double;                                \
  typedef className<float> className##_float;

#endif
