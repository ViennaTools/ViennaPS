#pragma once

#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <lsCalculateVisibilities.hpp>
#include <models/psDirectionalProcess.hpp>
#include <psMaterials.hpp>

#include <psRateGrid.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class velocityFieldFromFile : public VelocityField<NumericType, D> {
public:
  using Interpolation = typename RateGrid<NumericType, D>::Interpolation;

  velocityFieldFromFile(const std::string &ratesFile,
                        const Vec3D<NumericType> &dir,
                        const Vec2D<NumericType> &off,
                        const NumericType isoScale = 0.,
                        const NumericType dirScale = 1.,
                        const std::vector<Material> &masks = {Material::Mask},
                        bool calcVis = true)
      : direction(dir), offset(off), isotropicScale(isoScale),
        directionalScale(dirScale), maskMaterials(masks),
        calculateVisibility(calcVis &&
                            (dir[0] != 0. || dir[1] != 0. || dir[2] != 0.)) {

    if (!rateGrid.loadFromCSV(ratesFile)) {
      std::cerr << "Error: Failed to load rate grid from " << ratesFile
                << std::endl;
      return;
    }

    rateGrid.setOffset(off);
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> & /*normalVector*/,
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

  int getTranslationFieldOptions() const override { return 0; }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> /*velocities*/,
               const NumericType /*processTime*/) override {
    visibilities.clear();

    if (calculateVisibility) {
      std::string label = "Visibilities";
      auto surfaceLS = domain->getLevelSets().back();
      viennals::CalculateVisibilities<NumericType, D>(surfaceLS, direction,
                                                      label)
          .apply();
      visibilities = *surfaceLS->getPointData().getScalarData(label);
    }
  }

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

} // namespace viennaps
