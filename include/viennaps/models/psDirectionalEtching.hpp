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

public:
  DirectionalEtchVelocityField(Vec3D<NumericType> direction,
                               const NumericType directionalVelocity,
                               const NumericType isotropicVelocity,
                               const std::vector<int> &mask)
      : direction_(direction), directionalVelocity_(directionalVelocity),
        isotropicVelocity_(isotropicVelocity), maskMaterials_(mask) {}

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long) override {
    if (isMaskMaterial(material)) {
      return {0.};
    } else {
      auto rate = direction_;
      for (int i = 0; i < D; ++i) {
        if (rate[i] == 0.) {
          rate[i] -= isotropicVelocity_ * (normalVector[i] < 0 ? -1 : 1);
        } else {
          rate[i] *= directionalVelocity_;
        }
      }
      return rate;
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

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
template <typename NumericType, int D> class DirectionalEtching {
  SmartPointer<ProcessModel<NumericType, D>> processModel;

public:
  DirectionalEtching(const Vec3D<NumericType> &direction,
                     const NumericType directionalVelocity = 1.,
                     const NumericType isotropicVelocity = 0.,
                     const Material mask = Material::Mask) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();

    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt = {static_cast<int>(mask)};
    auto velField =
        SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity,
            maskMaterialsInt);

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->setProcessName("DirectionalEtching");
  }

  DirectionalEtching(const Vec3D<NumericType> &direction,
                     const NumericType directionalVelocity,
                     const NumericType isotropicVelocity,
                     const std::vector<Material> maskMaterials) {
    processModel = SmartPointer<ProcessModel<NumericType, D>>::New();

    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    std::vector<int> maskMaterialsInt;
    for (const auto &mat : maskMaterials) {
      maskMaterialsInt.push_back(static_cast<int>(mat));
    }
    // velocity field
    auto velField =
        SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity,
            maskMaterialsInt);

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->setProcessName("DirectionalEtching");
  }

  auto getProcessModel() const { return processModel; }
};

} // namespace viennaps
