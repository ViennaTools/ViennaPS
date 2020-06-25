#ifndef CS_CONVERSION_TABLE_HPP
#define CS_CONVERSION_TABLE_HPP

#include <cmath>
#include <limits>

namespace {
template <unsigned N_LS, unsigned N_X, unsigned N_Y>
class csConversionTableFactory {
  bool constexpr isZero(double number) const {
    if (number < 0.) {
      number -= number;
    }
    if (number < 1e-12) {
      return true;
    }
    return false;
  }

  double constexpr dotProduct(double a_x, double a_y, double a_z, double b_x,
                              double b_y, double b_z) const {
    return a_x * b_x + a_y * b_y + a_z * b_z;
  }

  double constexpr sqrtNewtonRaphson(double x, double current,
                                     double previous) const {
    return current == previous
               ? current
               : sqrtNewtonRaphson(x, 0.5 * (current + x / current), current);
  }

  double constexpr sqrt(double x) const {
    if (x >= 0 && x < std::numeric_limits<double>::infinity()) {
      return sqrtNewtonRaphson(x, x, 0.);
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  double constexpr fixLimit(double limit) const {
    if (limit < 0.) {
      return 0;
    } else if (limit > 1.) {
      return 1;
    } else {
      return limit;
    }
  }

  double constexpr calculateFillingFraction2D(double levelSetValue, double n_x,
                                              double n_y) const {
    double dot = dotProduct(0.5 - n_x * levelSetValue,
                            0.5 - n_y * levelSetValue, 0.0, n_x, n_y, 0.0);
    double a = fixLimit(dot / n_x);
    double b = fixLimit((dot - n_y) / n_x);

    return b + dot * (a - b) / n_y - n_x * (a * a - b * b) / (2 * n_y);
  }

  double constexpr calculateV1(double dot, double b, double n_x, double n_y,
                               double n_z) const {
    return (b * (dot - 0.5 * n_y) - 0.5 * n_x * b * b) / n_z;
  }

  double constexpr calculateV2(double dot, double a, double b, double n_x,
                               double n_y, double n_z) const {
    double a2 = a * a;
    double a3 = a2 * a;
    double b2 = b * b;
    double b3 = b2 * b;
    return (n_x * n_x * n_x * (a3 - b3) / 3.0 - dot * n_x * (a2 - b2) +
            dot * dot * (a - b)) /
           (2 * n_y * n_z);
  }

  double constexpr calculateFillingFraction(double levelSetValue, double n_x,
                                            double n_y, double n_z) const {
    // if it is axis parallel, just return the correct value depending on
    // dimension
    {
      bool x0 = isZero(n_x);
      bool y0 = isZero(n_y);
      bool z0 = isZero(n_z);
      if ((x0 && y0) || (y0 && z0) || (z0 && x0)) {
        return 0.5 - levelSetValue;
      }

      if (x0) {
        return calculateFillingFraction2D(levelSetValue, n_y, n_z);
      } else if (y0) {
        return calculateFillingFraction2D(levelSetValue, n_x, n_z);
      } else if (z0) {
        return calculateFillingFraction2D(levelSetValue, n_x, n_y);
      }
    }

    // if there is no axis alignment, just calculate the value
    double dot =
        dotProduct(0.5 - n_x * levelSetValue, 0.5 - n_y * levelSetValue,
                   0.5 - n_z * levelSetValue, n_x, n_y, n_z);
    double a = fixLimit(dot / n_x);
    double b = fixLimit((dot - n_y) / n_x);
    double c = fixLimit((dot - n_z) / n_x);
    double d = fixLimit((dot - n_y - n_z) / n_x);

    return calculateV1(dot, b, n_x, n_y, n_z) +
           calculateV2(dot, a, b, n_x, n_y, n_z) -
           calculateV1(dot, d, n_x, n_y, n_z) -
           calculateV2(dot, c, d, n_x, n_y, n_z) + 0.5 * n_z * (c - d) / n_y;
  }

public:
  constexpr csConversionTableFactory() {
    for (unsigned ls = 0; ls <= N_LS; ++ls) {
      double value = double(ls) / double(N_LS) - 0.5;
      for (unsigned n_x = 0; n_x <= N_X; ++n_x) {
        double x = double(n_x) / double(N_X);
        for (unsigned n_y = 0; n_y <= N_Y; ++n_y) {
          double y = double(n_y) / double(N_Y);
          double sum = x * x + y * y;
          double z = 0.0;
          if (sum < 1.0 - 1e-9) {
            z = sqrt(1.0 - sum);
          }
          double result = calculateFillingFraction(value, x, y, z);

          if (result < 0.0 || result > 1.0) {
            // if this is an error during compilation, the constexpr result was
            // wrong
            throw std::logic_error("FillingFraction must be [0, 1]!");
          }

          table[ls][n_x][n_y] = result;
        }
      }
    }
  }

  double table[N_LS + 1][N_X + 1][N_Y + 1];
};
} // namespace

class csConversionTable {
  inline static constexpr std::size_t numberOfLSValues = 100;
  inline static constexpr std::size_t numberOfNormalValues = 20;

  inline static constexpr auto factory =
      csConversionTableFactory<numberOfLSValues, numberOfNormalValues,
                               numberOfNormalValues>();
  // csConversionTableFactory<numberOfValues, numberOfValues, numberOfValues>
  // factory = csConversionTableFactory<numberOfValues, numberOfValues,
  // numberOfValues>();

  std::size_t lsValueToIndex(double value) const {
    return static_cast<std::size_t>(
        std::round((0.5 + value) * numberOfLSValues));
  }

  std::size_t normalValueToIndex(double value) const {
    return static_cast<std::size_t>(std::round(value * numberOfNormalValues));
  }

public:
  std::size_t getNumberOfLSValues() const { return numberOfLSValues; }

  std::size_t getNumberOfNormalValues() const { return numberOfNormalValues; }
  // double getLevelSetValue(double fillingFraction, double n_x,
  //                               double n_y) const {}

  double getFillingFraction(double levelSetValue, double n_x,
                            double n_y) const {
    auto ls = lsValueToIndex(levelSetValue);
    auto x = normalValueToIndex(n_x);
    auto y = normalValueToIndex(n_y);

    return factory.table[ls][x][y];
  }
};

#endif // CS_CONVERSION_TABLE_HPP