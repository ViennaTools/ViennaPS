#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

namespace DirectionalEtchingImplementation {
template <class NumericType, int D>
class DirectionalEtchVelocityField : public psVelocityField<NumericType> {
  const std::array<NumericType, 3> direction_;
  const NumericType directionalVelocity_;
  const NumericType isotropicVelocity_;
  const std::vector<int> maskMaterials_;

public:
  DirectionalEtchVelocityField(std::array<NumericType, 3> direction,
                               const NumericType directionalVelocity,
                               const NumericType isotropicVelocity,
                               const std::vector<int> &mask)
      : direction_(direction), directionalVelocity_(directionalVelocity),
        isotropicVelocity_(isotropicVelocity), maskMaterials_(mask) {}

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
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
} // namespace DirectionalEtchingImplementation

/// Directional etching with one masking material.
template <typename NumericType, int D>
class psDirectionalEtching : public psProcessModel<NumericType, D> {
public:
  psDirectionalEtching(const std::array<NumericType, 3> &direction,
                       const NumericType directionalVelocity = 1.,
                       const NumericType isotropicVelocity = 0.,
                       const psMaterial mask = psMaterial::Mask) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    std::vector<int> maskMaterialsInt = {static_cast<int>(mask)};
    auto velField = psSmartPointer<
        DirectionalEtchingImplementation::DirectionalEtchVelocityField<
            NumericType, D>>::New(direction, directionalVelocity,
                                  isotropicVelocity, maskMaterialsInt);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }

  psDirectionalEtching(const std::array<NumericType, 3> &direction,
                       const NumericType directionalVelocity,
                       const NumericType isotropicVelocity,
                       const std::vector<psMaterial> maskMaterials) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    std::vector<int> maskMaterialsInt;
    for (const auto &mat : maskMaterials) {
      maskMaterialsInt.push_back(static_cast<int>(mat));
    }
    // velocity field
    auto velField = psSmartPointer<
        DirectionalEtchingImplementation::DirectionalEtchVelocityField<
            NumericType, D>>::New(direction, directionalVelocity,
                                  isotropicVelocity, maskMaterialsInt);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }
};
