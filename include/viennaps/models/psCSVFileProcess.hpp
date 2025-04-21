#pragma once

#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include <psMaterials.hpp>

#include <lsCalculateVisibilities.hpp>

#include <models/psDirectionalProcess.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class velocityFieldFromFile : public VelocityField<NumericType, D> {
public:
  using OffsetType = std::array<NumericType, D == 2 ? 1 : 2>;
  enum class Interpolation { LINEAR, IDW, CUSTOM };

  velocityFieldFromFile(const std::string &ratesFile,
                        const Vec3D<NumericType> &dir, const OffsetType &off,
                        const NumericType isoScale = 0.,
                        const NumericType dirScale = 1.,
                        const std::vector<Material> &masks =
                            std::vector<Material>{
                                Material::Mask}, // Default to Material::Mask)
                        bool calcVis = true)
      : direction(dir), offset(off), isotropicScale(isoScale),
        directionalScale(dirScale), maskMaterials(masks),
        calculateVisibility(calcVis &&
                            (dir[0] != 0. || dir[1] != 0. || dir[2] != 0.)) {
    ratePoints = readRateCSV(ratesFile);
    if (ratePoints.empty()) {
      std::cerr << "Error: No data in rate CSV file: " << ratesFile
                << std::endl;
      return;
    }
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {

    if (isMaskMaterial(material, maskMaterials)) {
      return 0.;
    }

    if (calculateVisibility && (pointId >= visibilities_.size() || visibilities_.at(pointId) == 0.)) {
      return 0.;
    }

    // if (calculateVisibility &&
    //   pointId < visibilities_.size() &&
    //   visibilities_.at(pointId) == 0.) {
    //     return 0.;
    // }

    NumericType scalarVelocity = interpolateRate(coordinate);
    return scalarVelocity * isotropicScale;
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    Vec3D<NumericType> vectorVelocity{0., 0., 0.};

    if (isMaskMaterial(material, maskMaterials)) {
      return Vec3D<NumericType>{0., 0., 0.};
    }

    if (calculateVisibility && (pointId >= visibilities_.size() || visibilities_.at(pointId) == 0.)) {
      return Vec3D<NumericType>{0., 0., 0.};
    }

    // if (calculateVisibility &&
    //   pointId < visibilities_.size() &&
    //   visibilities_.at(pointId) == 0.) {
    //     return Vec3D<NumericType>{0., 0., 0.};
    // }

    NumericType scaling = interpolateRate(coordinate);
    Vec3D<NumericType> potentialVelocity =
        direction * scaling * directionalScale;

    NumericType dotProduct = DotProduct(potentialVelocity, normalVector);
    if (dotProduct != 0) {
      vectorVelocity = vectorVelocity - potentialVelocity;
    }

    return vectorVelocity;
  }

  // The translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType processTime) override {

    visibilities_.clear();

    // Calculate visibilities
    auto surfaceLS = domain->getLevelSets().back();
    if (calculateVisibility) {
      std::string label = "Visibilities";
      viennals::CalculateVisibilities<NumericType, D>(surfaceLS, direction,
                                                      label)
          .apply();
      visibilities_ = *surfaceLS->getPointData().getScalarData(label);
    }
  }

  void setRateProfile(std::vector<std::array<NumericType, D>> points) {
    ratePoints = std::move(points);
  }

  void setOffset(std::array<NumericType, D> off) { offset = off; }

  void setInterpolationMode(Interpolation mode) { interpolationMode = mode; }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    customInterpolator = func;
  }

protected:
  static bool isMaskMaterial(const int material,
                             const std::vector<Material> &maskMaterials) {
    for (const auto &maskMaterial : maskMaterials) {
      if (MaterialMap::isMaterial(material, maskMaterial)) {
        return true;
      }
    }
    return false;
  }

private:
  std::vector<std::array<NumericType, D>> ratePoints;
  OffsetType offset;
  Vec3D<NumericType> direction;
  bool calculateVisibility;
  std::vector<NumericType> visibilities_;
  std::vector<Material> maskMaterials;
  NumericType isotropicScale;
  NumericType directionalScale;
  std::optional<Interpolation> interpolationMode;
  std::function<NumericType(const Vec3D<NumericType> &)> customInterpolator;

  std::vector<std::array<NumericType, D>>
  readRateCSV(const std::string &filename) {
    std::vector<std::array<NumericType, D>> result;
  
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open CSV rates file: " << filename << std::endl;
      return result;
    }
  
    std::string line;
    bool headerSkipped = false;
  
    while (std::getline(file, line)) {
      // Remove possible Windows CR characters
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
      file >> std::noskip;
  
      if (!headerSkipped) {
        headerSkipped = true;
        continue;
      }
  
      std::istringstream stream(line);
      std::string cell;
      std::array<NumericType, D> entry{};
      int i = 0;
  
      try {
        // Read coordinate columns
        while (std::getline(stream, cell, ',') && i < D - 1) {
          entry[i++] = static_cast<NumericType>(std::stod(cell));
        }
  
        // Last value is the rate
        if (std::getline(stream, cell, ',')) {
          entry[D - 1] = static_cast<NumericType>(std::stod(cell));
        } else {
          std::cerr << "Warning: Missing rate value in line: " << line << std::endl;
          continue;
        }
  
        result.push_back(entry);
  
      } catch (const std::exception &e) {
        std::cerr << "CSV parse error in file '" << filename << "': " << e.what()
                  << "\n  Offending line: " << line << std::endl;
        continue;
      }
    }
  
    if (result.empty()) {
      std::cerr << "Warning: CSV file read but no data points loaded from " << filename << std::endl;
    }
    
    interpolationMode = detectInterpolationMode(result);
    file.close();
    return result;
  }
  
  Interpolation detectInterpolationMode(
      const std::vector<std::array<NumericType, D>> &points) const {
    bool isStructured = true;

    if constexpr (D == 2) {
      for (std::size_t i = 1; i < points.size(); ++i) {
        if (points[i][0] <= points[i - 1][0]) {
          isStructured = false;
          break;
        }
      }
    } else if constexpr (D == 3) {
      std::set<NumericType> xSet, ySet;
      for (const auto &pt : points) {
        xSet.insert(pt[0]);
        ySet.insert(pt[1]);
      }

      std::size_t expected = xSet.size() * ySet.size();
      if (expected != points.size()) {
        isStructured = false;
      }
    }

    return isStructured ? Interpolation::LINEAR : Interpolation::IDW;
  }

  NumericType interpolateRateLinear(const Vec3D<NumericType> &coord) const {
    if (ratePoints.empty())
      return 1.0;

    Vec3D<NumericType> globalCoord = coord;
    for (int i = 0; i < D; ++i)
      globalCoord[i] += offset[i];

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

      // Find the four surrounding points: (x0, y0), (x1, y0), (x0, y1), (x1,
      // y1)
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

      if (x0It == xVals.begin() || y0It == yVals.begin())
        return 1.0;
      if (x0It == xVals.end())
        --x0It;
      if (y0It == yVals.end())
        --y0It;

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
      if (denom == 0.)
        return 1.0;

      NumericType rate = 1.0;
      rate = 1.0 / denom *
             (q11 * (x1 - x) * (y1 - y) + q21 * (x - x0) * (y1 - y) +
              q12 * (x1 - x) * (y - y0) + q22 * (x - x0) * (y - y0));

      return rate;
    }
  }

  NumericType interpolateRateIDW(const Vec3D<NumericType> &coord) const {
    if (ratePoints.empty())
      return 1.0;

    Vec3D<NumericType> globalCoord = coord;
    for (int i = 0; i < D - 1; ++i)
      globalCoord[i] += offset[i];

    const NumericType epsilon = 1e-6;
    const int k = 4; // number of nearest neighbors to consider

    // Find k nearest neighbors
    std::vector<std::pair<NumericType, NumericType>> distances; // (dist, rate)
    for (const auto &pt : ratePoints) {
      NumericType dist = 0.0;
      for (int i = 0; i < D - 1; ++i)
        dist += std::pow(globalCoord[i] - pt[i], 2);
      dist = std::sqrt(dist);
      distances.emplace_back(dist, pt[D - 1]);
    }

    std::partial_sort(distances.begin(), distances.begin() + k,
                      distances.end());

    NumericType weightSum = 0.0;
    NumericType weightedRateSum = 0.0;

    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
      auto [dist, rate] = distances[i];
      if (dist < epsilon)
        return rate; // Avoid divide-by-zero
      NumericType weight = 1.0 / (dist * dist);
      weightSum += weight;
      weightedRateSum += weight * rate;
    }

    return weightedRateSum / weightSum;
  }

  NumericType interpolateRate(const Vec3D<NumericType> &coord) const {
    if (interpolationMode == Interpolation::LINEAR) {
      return interpolateRateLinear(coord);
    } else if (interpolationMode == Interpolation::IDW) {
      return interpolateRateIDW(coord);
    } else if (interpolationMode == Interpolation::CUSTOM) {
      return customInterpolator(coord);
    } else {
      std::cerr << "Error: Invalid interpolation mode." << std::endl;
      return 1.0;
    }
  }
};

} // namespace impl

/// Rate determined by CSV file.
template <typename NumericType, int D>
class CSVFileProcess : public ProcessModel<NumericType, D> {
  using OffsetType = std::array<NumericType, D == 2 ? 1 : 2>;

public:
  CSVFileProcess(const std::string &ratesFile, const Vec3D<NumericType> &dir,
                 const OffsetType &off, const NumericType isoScale = 0.,
                 const NumericType dirScale = 1.,
                 const std::vector<Material> &masks =
                     std::vector<Material>{
                         Material::Mask}, // Default to Material::Mask)
                 bool calcVis = true) {
    auto velField =
        SmartPointer<impl::velocityFieldFromFile<NumericType, D>>::New(
            ratesFile, dir, off, isoScale, dirScale, masks, calcVis);
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("CSVFileProcess");
  }

  void setInterpolationMode(
      typename impl::velocityFieldFromFile<NumericType, D>::Interpolation
          mode) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setInterpolationMode(mode);
  }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setCustomInterpolator(func);
  }
};

} // namespace viennaps
