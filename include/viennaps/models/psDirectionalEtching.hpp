#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <vcVectorUtil.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <class NumericType, int D>
class DirectionalEtchVelocityField : public VelocityField<NumericType> {
  const Vec3D<NumericType> direction_;
  const NumericType directionalVelocity_;
  const NumericType isotropicVelocity_;
  const std::vector<int> maskMaterials_;
  const bool useVisibilities_;

public:
  DirectionalEtchVelocityField(Vec3D<NumericType> direction,
                               const NumericType directionalVelocity,
                               const NumericType isotropicVelocity,
                               const std::vector<int> &mask,
                               const bool useVisibilities = false)
      : direction_(direction), directionalVelocity_(directionalVelocity),
        isotropicVelocity_(isotropicVelocity), maskMaterials_(mask),
        useVisibilities_(useVisibilities) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    if (isMaskMaterial(material)) {
      return 0.;
    } else {
      return -isotropicVelocity_;
    }
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    if (isMaskMaterial(material)) {
      return {0.};
    } else if (useVisibilities_ && this->visibilities_->at(pointId) == 0.) {
      return {0.};
    } else {
      return direction_ * directionalVelocity_;
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

  bool useVisibilities() const override { return useVisibilities_; }

private:
  bool isMaskMaterial(const int material) const {
    for (const auto &mat : maskMaterials_) {
      if (material == mat)
        return true;
    }
    return false;
  }
};
} // namespace impl

/// Directional etching with one masking material.
template <typename NumericType, int D>
class DirectionalEtching : public ProcessModel<NumericType, D> {
public:
  DirectionalEtching(const Vec3D<NumericType> &direction,
                     const NumericType directionalVelocity = 1.,
                     const NumericType isotropicVelocity = 0.,
                     const bool useVisibilities = false,
                     const Material mask = Material::Mask) {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt = {static_cast<int>(mask)};
    auto velField =
        SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity, maskMaterialsInt,
            useVisibilities);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }

  DirectionalEtching(const Vec3D<NumericType> &direction,
                     const NumericType directionalVelocity,
                     const NumericType isotropicVelocity,
                     const bool useVisibilities,
                     const std::vector<Material> maskMaterials) {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    std::vector<int> maskMaterialsInt;
    for (const auto &mat : maskMaterials) {
      maskMaterialsInt.push_back(static_cast<int>(mat));
    }
    // velocity field
    auto velField =
        SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity, maskMaterialsInt,
            useVisibilities);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }
};

} // namespace viennaps
