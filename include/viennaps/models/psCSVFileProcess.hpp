#pragma once

#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../psDomain.hpp"
#include "../psMaterials.hpp"
#include "../psRateGrid.hpp"

#include <lsCalculateVisibilities.hpp>

#include <functional>
#include <string>
#include <vector>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class VelocityFieldFromFile : public VelocityField<NumericType, D> {
public:
  using Interpolation = typename RateGrid<NumericType, D>::Interpolation;

  VelocityFieldFromFile(const std::string &ratesFile,
                        const Vec3D<NumericType> &dir,
                        const Vec2D<NumericType> &off,
                        const NumericType isoScale = 0.,
                        const NumericType dirScale = 1.,
                        const std::vector<Material> &masks = {Material::Mask},
                        bool calcVis = true)
      : offset(off), direction(dir),
        calculateVisibility(calcVis &&
                            (dir[0] != 0. || dir[1] != 0. || dir[2] != 0.)),
        maskMaterials(masks), isotropicScale(isoScale),
        directionalScale(dirScale) {

    if (!rateGrid.loadFromCSV(ratesFile)) {
      std::cerr << "Error: Failed to load rate grid from " << ratesFile
                << std::endl;
      return;
    }

    rateGrid.setOffset(off);
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    if (isMaskMaterial(material))
      return 0.;

    if (calculateVisibility &&
        (pointId >= visibilities.size() || visibilities[pointId] == 0.))
      return 0.;

    return rateGrid.interpolate(coordinate) * isotropicScale;
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    if (isMaskMaterial(material))
      return {0., 0., 0.};

    if (calculateVisibility &&
        (pointId >= visibilities.size() || visibilities[pointId] == 0.))
      return {0., 0., 0.};

    auto potentialVelocity =
        direction * rateGrid.interpolate(coordinate) * directionalScale;

    if (DotProduct(potentialVelocity, normalVector) != 0.) {
      for (int i = 0; i < 3; ++i)
        potentialVelocity[i] = -potentialVelocity[i];
      return potentialVelocity;
    }

    return {0., 0., 0.};
  }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType processTime) override {
    visibilities.clear();

    auto surfaceLS = domain->getLevelSets().back();
    if (calculateVisibility) {
      std::string label = "Visibilities";
      viennals::CalculateVisibilities<NumericType, D>(surfaceLS, direction,
                                                      label)
          .apply();
      visibilities = *surfaceLS->getPointData().getScalarData(label);
    }
  }

  void setIDWNeighbors(const int k = 4) { rateGrid.setIDWNeighbors(k); }

  void setInterpolationMode(const std::string &str) {
    rateGrid.setInterpolationMode(rateGrid.fromString(str));
  }

  void setInterpolationMode(Interpolation mode) {
    rateGrid.setInterpolationMode(mode);
  }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    rateGrid.setCustomInterpolator(func);
  }

  void setOffset(const Vec2D<NumericType> &off) {
    offset = off;
    rateGrid.setOffset({off[0], off[1]});
  }

private:
  Vec2D<NumericType> offset;
  Vec3D<NumericType> direction;
  bool calculateVisibility;
  std::vector<NumericType> visibilities;
  std::vector<Material> maskMaterials;
  NumericType isotropicScale;
  NumericType directionalScale;

  RateGrid<NumericType, D> rateGrid;

  bool isMaskMaterial(const int material) const {
    for (const auto &mask : maskMaterials)
      if (MaterialMap::isMaterial(material, mask))
        return true;
    return false;
  }
};

} // namespace impl

template <typename NumericType, int D>
class CSVFileProcess : public ProcessModelCPU<NumericType, D> {
public:
  CSVFileProcess(const std::string &ratesFile, const Vec3D<NumericType> &dir,
                 const Vec2D<NumericType> &off, NumericType isoScale = 0.,
                 NumericType dirScale = 1.,
                 const std::vector<Material> &masks = {Material::Mask},
                 bool calcVis = true) {

    auto normalizeVec3D =
        [](const Vec3D<NumericType> &v) -> Vec3D<NumericType> {
      NumericType norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (norm == 0)
        throw std::runtime_error("Cannot normalize zero-length vector");
      return {v[0] / norm, v[1] / norm, v[2] / norm};
    };

    Vec3D<NumericType> normDir = normalizeVec3D(dir);

    auto velField =
        SmartPointer<impl::VelocityFieldFromFile<NumericType, D>>::New(
            ratesFile, normDir, off, isoScale, dirScale, masks, calcVis);

    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("CSVFileProcess");
  }

  void setIDWNeighbors(const int k = 4) {
    auto velField =
        std::dynamic_pointer_cast<impl::VelocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setIDWNeighbors(k);
  }

  void setInterpolationMode(const std::string &str) {
    auto velField =
        std::dynamic_pointer_cast<impl::VelocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setInterpolationMode(str);
  }

  void setInterpolationMode(
      typename impl::VelocityFieldFromFile<NumericType, D>::Interpolation
          mode) {
    auto velField =
        std::dynamic_pointer_cast<impl::VelocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setInterpolationMode(mode);
  }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    auto velField =
        std::dynamic_pointer_cast<impl::VelocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setCustomInterpolator(func);
  }

  void setOffset(const Vec2D<NumericType> &off) {
    auto velField =
        std::dynamic_pointer_cast<impl::VelocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setOffset(off);
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(CSVFileProcess)

} // namespace viennaps
