#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>

#include <compact/psCSVReader.hpp>

namespace viennaps {

using namespace viennacore;

template <typename T> struct Vec2DHash {
  std::size_t operator()(const Vec2D<T> &v) const {
    auto h1 = std::hash<T>{}(v[0]);
    auto h2 = std::hash<T>{}(v[1]);
    return h1 ^ (h2 << 1);
  }
};

template <typename NumericType, int D> class RateGrid {
public:
  enum class Interpolation { LINEAR, IDW, CUSTOM };

  bool loadFromCSV(const std::string &filename) {
    CSVReader<NumericType> reader(filename);
    auto content = reader.readContent();
    if (!content.has_value()) {
      std::cerr << "RateGrid: Failed to read CSV content from " << filename
                << "\n";
      return false;
    }

    const auto &rawData = content.value();
    points.clear();
    rateMap.clear();
    xVals.clear();
    yVals.clear();

    for (const auto &row : rawData) {
      if (row.size() != D) {
        Logger::getInstance()
            .addWarning("RateGrid: Invalid number of columns in row!")
            .print();
        continue;
      }

      std::array<NumericType, D> pt{};
      for (int i = 0; i < D; ++i)
        pt[i] = row[i];

      points.push_back(pt);

      if constexpr (D == 2) {
        xVals.insert(pt[0]);
      } else if constexpr (D == 3) {
        xVals.insert(pt[0]);
        yVals.insert(pt[1]);
        rateMap[{pt[0], pt[1]}] = pt[2];
      }
    }

    interpolationMode = detectInterpolationMode();
    return true;
  }

  void setOffset(const Vec2D<NumericType> &off) { offset = off; }

  static Interpolation fromString(const std::string &str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "linear")
      return Interpolation::LINEAR;
    if (s == "idw")
      return Interpolation::IDW;
    if (s == "custom")
      return Interpolation::CUSTOM;
    throw std::invalid_argument("Unknown interpolation mode: " + str);
  }

  void setInterpolationMode(Interpolation mode) { interpolationMode = mode; }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    customInterpolator = func;
    interpolationMode = Interpolation::CUSTOM;
  }

  NumericType interpolate(const Vec3D<NumericType> &coord) const {
    if (!interpolationMode.has_value()) {
      std::cerr << "Interpolation mode not set." << std::endl;
      return 1.0;
    }

    if (interpolationMode == Interpolation::LINEAR)
      return interpolateLinear(coord);
    else if (interpolationMode == Interpolation::IDW)
      return interpolateIDW(coord);
    else if (interpolationMode == Interpolation::CUSTOM)
      return customInterpolator(coord);

    return 1.0;
  }

private:
  std::vector<std::array<NumericType, D>> points;
  std::unordered_map<Vec2D<NumericType>, NumericType, Vec2DHash<NumericType>>
      rateMap;
  std::set<NumericType> xVals, yVals;

  Vec2D<NumericType> offset{};
  std::optional<Interpolation> interpolationMode;
  std::function<NumericType(const Vec3D<NumericType> &)> customInterpolator;

  Interpolation detectInterpolationMode() const {
    if constexpr (D == 2) {
      for (std::size_t i = 1; i < points.size(); ++i) {
        if (points[i][0] <= points[i - 1][0]) {
          return Interpolation::IDW;
        }
      }
      return Interpolation::LINEAR;
    } else if constexpr (D == 3) {
      std::size_t expected = xVals.size() * yVals.size();
      return (expected == rateMap.size()) ? Interpolation::LINEAR
                                          : Interpolation::IDW;
    }
    return Interpolation::IDW;
  }

  NumericType interpolateLinear(const Vec3D<NumericType> &coord) const {
    if constexpr (D == 2) {
      NumericType x = coord[0] + offset[0];
      for (std::size_t i = 1; i < points.size(); ++i) {
        if (x < points[i][0]) {
          NumericType x0 = points[i - 1][0];
          NumericType r0 = points[i - 1][1];
          NumericType x1 = points[i][0];
          NumericType r1 = points[i][1];
          return r0 + ((x - x0) / (x1 - x0)) * (r1 - r0);
        }
      }
      return points.back()[1]; // extrapolate
    } else if constexpr (D == 3) {
      NumericType x = coord[0] + offset[0];
      NumericType y = coord[1] + offset[1];

      auto x0It = xVals.lower_bound(x);
      auto y0It = yVals.lower_bound(y);

      if (x0It == xVals.begin() || y0It == yVals.begin())
        return 1.0;

      if (x0It == xVals.end())
        --x0It;
      if (y0It == yVals.end())
        --y0It;

      auto x1It = x0It;
      auto y1It = y0It;
      --x0It;
      --y0It;

      NumericType x0 = *x0It, x1 = *x1It;
      NumericType y0 = *y0It, y1 = *y1It;

      auto safeGet = [&](NumericType xi, NumericType yi) {
        auto it = rateMap.find({xi, yi});
        if (it != rateMap.end())
          return it->second;
        std::cerr << "Missing rate at (" << xi << ", " << yi << ")\n";
        return NumericType(1.0);
      };

      NumericType q11 = safeGet(x0, y0);
      NumericType q21 = safeGet(x1, y0);
      NumericType q12 = safeGet(x0, y1);
      NumericType q22 = safeGet(x1, y1);

      NumericType denom = (x1 - x0) * (y1 - y0);
      if (denom == 0.0)
        return 1.0;

      NumericType rate =
          1.0 / denom *
          (q11 * (x1 - x) * (y1 - y) + q21 * (x - x0) * (y1 - y) +
           q12 * (x1 - x) * (y - y0) + q22 * (x - x0) * (y - y0));

      return rate;
    }
    return 1.0;
  }

  NumericType interpolateIDW(const Vec3D<NumericType> &coord) const {
    if (points.empty())
      return 1.0;

    Vec2D<NumericType> query;
    for (int i = 0; i < D - 1; ++i)
      query[i] = coord[i] + offset[i];

    const NumericType epsilon = 1e-6;
    const int k = 4;

    std::vector<std::pair<NumericType, NumericType>> dists;
    for (const auto &pt : points) {
      NumericType dist = 0.0;
      for (int i = 0; i < D - 1; ++i)
        dist += std::pow(pt[i] - query[i], 2);
      dist = std::sqrt(dist);
      dists.emplace_back(dist, pt[D - 1]);
    }

    std::partial_sort(dists.begin(),
                      dists.begin() + std::min(k, (int)dists.size()),
                      dists.end());

    NumericType wSum = 0.0;
    NumericType wrSum = 0.0;
    for (int i = 0; i < std::min(k, (int)dists.size()); ++i) {
      auto [dist, val] = dists[i];
      if (dist < epsilon)
        return val;
      NumericType w = 1.0 / (dist * dist);
      wSum += w;
      wrSum += w * val;
    }

    return (wSum > 0) ? wrSum / wSum : 1.0;
  }
};

} // namespace viennaps
