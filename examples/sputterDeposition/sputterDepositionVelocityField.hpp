#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include <models/psDirectionalProcess.hpp>

namespace viennaps {

template <typename NumericType, int D>
class sputterDepositionVelocityField
    : public impl::DirectionalVelocityField<NumericType, D> {
public:
  using RateSet = impl::RateSet<NumericType>;
  using Base = impl::DirectionalVelocityField<NumericType, D>;
  using TrenchCenterType = std::array<NumericType, D == 2 ? 1 : 2>;

  sputterDepositionVelocityField(
    const RateSet &rateInfo,
    const std::string &rateFile,
    const TrenchCenterType &trenchCenter_)
    : Base(std::vector<RateSet>{rateInfo}),
      trenchCenter(trenchCenter_) {
      ratePoints = readRateCSV(rateFile);
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    Vec3D<NumericType> vectorVelocity{0., 0., 0.};

    const auto &rateInfo = this->getRateSets();
    const auto &vis = this->getVisibilities();

    for (unsigned rateSetID = 0; rateSetID < rateInfo.size(); ++rateSetID) {
      const auto &rateSet = rateInfo[rateSetID];
      if (Base::isMaskMaterial(material, rateSet.maskMaterials)) continue;

      if (rateSet.calculateVisibility && vis.at(rateSetID).at(pointId) == 0.)
        continue;

      NumericType scaling = interpolateRateLinear(coordinate);
      Vec3D<NumericType> potentialVelocity =
          rateSet.direction * scaling;

      NumericType dotProduct = DotProduct(potentialVelocity, normalVector);
      if (dotProduct != 0) {
        vectorVelocity = vectorVelocity - potentialVelocity;
      }
    }
    return vectorVelocity;
  }

  void setRateProfile(std::vector<std::array<NumericType, D>> points) {
    ratePoints = std::move(points);
  }

  void setTrenchCenter(std::array<NumericType, D> center) {
    trenchCenter = center;
  }

private:
  std::vector<std::array<NumericType, D>> ratePoints;
  TrenchCenterType trenchCenter;

  std::vector<std::array<NumericType, D>> readRateCSV(const std::string &filename) {
    std::vector<std::array<NumericType, D>> result;
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open rate CSV file: " << filename << std::endl;
      return result;
    }
  
    std::string line;
    bool headerSkipped = false;
  
    while (std::getline(file, line)) {
      if (!headerSkipped) {
        headerSkipped = true;
        continue;
      }
  
      std::istringstream stream(line);
      std::string cell;
      std::array<NumericType, D> entry;
  
      int i = 0;
      while (std::getline(stream, cell, ',') && i < D) {
        entry[i++] = static_cast<NumericType>(std::stod(cell));
      }
  
      // Last value is the rate
      if (std::getline(stream, cell, ',')) {
        if constexpr (D == 2)
          entry[1] = static_cast<NumericType>(std::stod(cell)); // x, rate
        else if constexpr (D == 3)
          entry[2] = static_cast<NumericType>(std::stod(cell)); // x, y, rate
      }
      result.push_back(entry);
    }
  
    return result;
  }

  NumericType interpolateRateLinear(const Vec3D<NumericType>& coord) const {
    if (ratePoints.empty())
      return 1.0;

    Vec3D<NumericType> globalCoord = coord;
    for (int i = 0; i < D; ++i)
      globalCoord[i] += trenchCenter[i];  
    

    if constexpr (D == 2) {
      // Linear interpolation along x
      NumericType x = globalCoord[0];
      for (std::size_t i = 1; i < ratePoints.size(); ++i) {
        if (x < ratePoints[i][0]) {
          NumericType x0 = ratePoints[i - 1][0];
          NumericType r0 = ratePoints[i - 1][1];
          NumericType x1 = ratePoints[i][0];
          NumericType r1 = ratePoints[i][1];
          NumericType t = (x - x0) / (x1 - x0);
          return r0 + t * (r1 - r0);
        }
      }
      return ratePoints.back()[1]; // extrapolate
    } else if constexpr (D == 3) {
      NumericType x = globalCoord[0];
      NumericType y = globalCoord[1];
    
      // Find the four surrounding points: (x0, y0), (x1, y0), (x0, y1), (x1, y1)
      std::map<std::pair<NumericType, NumericType>, NumericType> rateMap;
      for (const auto &pt : ratePoints) {
        rateMap[{pt[0], pt[1]}] = pt[2];
      }
    
      // Extract unique sorted x and y coords
      std::set<NumericType> xVals, yVals;
      for (const auto &[xy, _] : rateMap) {
        xVals.insert(xy.first);
        yVals.insert(xy.second);
      }
    
      auto x0It = xVals.lower_bound(x);
      auto y0It = yVals.lower_bound(y);
    
      if (x0It == xVals.begin() || y0It == yVals.begin()) return 1.0;
      if (x0It == xVals.end()) --x0It;
      if (y0It == yVals.end()) --y0It;
    
      auto x1It = x0It, y1It = y0It;
      --x0It;
      --y0It;
    
      NumericType x0 = *x0It, x1 = *x1It;
      NumericType y0 = *y0It, y1 = *y1It;
    
      NumericType q11 = rateMap[{x0, y0}];
      NumericType q21 = rateMap[{x1, y0}];
      NumericType q12 = rateMap[{x0, y1}];
      NumericType q22 = rateMap[{x1, y1}];
    
      // Bilinear interpolation
      NumericType denom = (x1 - x0) * (y1 - y0);
      if (denom == 0.) return 1.0;
    
      NumericType rate = 1.0;
      rate = 1.0 / denom * (
        q11 * (x1 - x) * (y1 - y) +
        q21 * (x - x0) * (y1 - y) +
        q12 * (x1 - x) * (y - y0) +
        q22 * (x - x0) * (y - y0)
      );
    
      return rate;
    }
  }

  NumericType interpolateRateIDW(const Vec3D<NumericType>& coord) const {
    if (ratePoints.empty())
      return 1.0;
  
    Vec3D<NumericType> globalCoord = coord;
    for (int i = 0; i < D - 1; ++i)
      globalCoord[i] += trenchCenter[i];
  
    const NumericType epsilon = 1e-6;
    const int k = 4; // number of nearest neighbors to consider
  
    // Find k nearest neighbors
    std::vector<std::pair<NumericType, NumericType>> distances; // (dist, rate)
    for (const auto& pt : ratePoints) {
      NumericType dist = 0.0;
      for (int i = 0; i < D - 1; ++i)
        dist += std::pow(globalCoord[i] - pt[i], 2);
      dist = std::sqrt(dist);
      distances.emplace_back(dist, pt[D - 1]);
    }
  
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
  
    NumericType weightSum = 0.0;
    NumericType weightedRateSum = 0.0;
  
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
      auto [dist, rate] = distances[i];
      if (dist < epsilon) return rate; // Avoid divide-by-zero
      NumericType weight = 1.0 / (dist * dist);
      weightSum += weight;
      weightedRateSum += weight * rate;
    }
  
    return weightedRateSum / weightSum;
  }
};

} // namespace viennaps
