#pragma once

#include <cmath>
#include <random>

#include <vcRNG.hpp>

namespace viennaps::impl {

using namespace viennacore;

inline double erfcinvApprox(double y) {
  // Approx by Mike Giles, accurate ~1e-9 in double. Domain y in (0,2).
  // See M. Giles, “Approximating the erfinv function.” In  GPU Computing Gems
  // Jade Edition, pp. 109-116. 2011.
  double w = -std::log((2.0 - y) * y);
  double p;
  if (w < 5.0) {
    w = w - 2.5;
    p = 2.81022636e-08;
    p = 3.43273939e-07 + p * w;
    p = -3.5233877e-06 + p * w;
    p = -4.39150654e-06 + p * w;
    p = 0.00021858087 + p * w;
    p = -0.00125372503 + p * w;
    p = -0.00417768164 + p * w;
    p = 0.246640727 + p * w;
    p = 1.50140941 + p * w;
  } else {
    w = std::sqrt(w) - 3.0;
    p = -0.000200214257;
    p = 0.000100950558 + p * w;
    p = 0.00134934322 + p * w;
    p = -0.00367342844 + p * w;
    p = 0.00573950773 + p * w;
    p = -0.0076224613 + p * w;
    p = 0.00943887047 + p * w;
    p = 1.00167406 + p * w;
    p = 2.83297682 + p * w;
  }
  return p;
}

inline double norm_inv_from_cdf(double p) {
  // Φ^{-1}(p) = -√2 * erfcinv(2p)
  return std::clamp(-1.4142135623730951 * erfcinvApprox(2.0 * p), 0.0, 1.0);
}

inline double Phi_from_x(double x) {
  // Φ(x) = 0.5 * erfc(-x/√2)
  return 0.5 * std::erfc(-x * 0.7071067811865476);
}

template <class NumericType>
inline NumericType sampleTruncatedNormal(RNG &rng, NumericType mu,
                                         NumericType sigma, NumericType L,
                                         NumericType U) {
  if (sigma <= NumericType(0))
    return std::min(std::max(mu, L), U);

  const double a = (double(L) - double(mu)) / double(sigma);
  const double b = (double(U) - double(mu)) / double(sigma);
  const double Fa = Phi_from_x(a);
  const double Fb = Phi_from_x(b);
  double w = Fb - Fa;

  if (w <= 1e-12) {
    const double pmid = Fa + 0.5 * w;
    return NumericType(double(mu) + double(sigma) * norm_inv_from_cdf(pmid));
  }

  std::uniform_real_distribution<double> uni(Fa, Fb);
  const double p = uni(rng);
  const double z = norm_inv_from_cdf(p);
  return NumericType(double(mu) + double(sigma) * z);
}

template <class NumericType>
inline NumericType updateEnergy(RNG &rng, NumericType E, NumericType incAngle,
                                NumericType A_energy,
                                const NumericType inflectAngle,
                                const NumericType n_l) {
  // Small incident angles are reflected with the energy fraction centered at
  // 0
  NumericType Eref_peak;
  if (incAngle >= inflectAngle) {
    Eref_peak = NumericType(1) - (NumericType(1) - A_energy) *
                                     (NumericType(M_PI_2) - incAngle) /
                                     (NumericType(M_PI_2) - inflectAngle);
  } else {
    Eref_peak = A_energy * std::pow(incAngle / inflectAngle, n_l);
  }
  // Normal distribution around the Eref_peak scaled by the particle energy
  const NumericType mu = E * Eref_peak;
  const NumericType sigma = NumericType(0.1) * E;
  return sampleTruncatedNormal(rng, mu, sigma, NumericType(0) /*lower bound*/,
                               E /*upper bound*/);
}

template <typename T>
inline T initNormalDistEnergy(RNG &rng, T mean, T sigma, T threshold) {
  if (sigma <= T(0))
    return mean;

  const T a = (threshold - mean) / sigma;
  const T Phi_a = Phi_from_x(a);

  std::uniform_real_distribution<T> uni(T(0), T(1));
  const T u = uni(rng);
  const T up = Phi_a + (T(1) - Phi_a) * u;

  const T z = norm_inv_from_cdf(up);
  return mean + sigma * z;
}

} // namespace viennaps::impl