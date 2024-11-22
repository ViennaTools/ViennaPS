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
                               const bool useVisibilities)
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
                     const Material mask = Material::Mask)
      : direction_(direction), directionalVelocity_(directionalVelocity),
        isotropicVelocity_(isotropicVelocity) {
    if (mask != Material::None)
      maskMaterials_.push_back(static_cast<int>(mask));
    initialize(direction_, directionalVelocity_, isotropicVelocity_, true,
               maskMaterials_);
  }

  DirectionalEtching(const Vec3D<NumericType> &direction,
                     const NumericType directionalVelocity,
                     const NumericType isotropicVelocity,
                     const std::vector<Material> maskMaterials)
      : direction_(direction), directionalVelocity_(directionalVelocity),
        isotropicVelocity_(isotropicVelocity) {
    for (const auto &mat : maskMaterials) {
      maskMaterials_.push_back(static_cast<int>(mat));
    }
    initialize(direction_, directionalVelocity_, isotropicVelocity_, true,
               maskMaterials_);
  }

  void disableVisibilityCheck() {
    initialize(direction_, directionalVelocity_, isotropicVelocity_, false,
               maskMaterials_);
  }

private:
  void initialize(const Vec3D<NumericType> &direction,
                  const NumericType directionalVelocity,
                  const NumericType isotropicVelocity,
                  const bool useVisibilities,
                  const std::vector<int> &maskMaterials) {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity, maskMaterials,
            useVisibilities);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }

private:
  Vec3D<NumericType> direction_;
  NumericType directionalVelocity_;
  NumericType isotropicVelocity_;
  std::vector<int> maskMaterials_;
};

} // namespace viennaps
